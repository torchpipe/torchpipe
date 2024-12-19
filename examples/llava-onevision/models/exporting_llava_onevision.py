import requests
from PIL import Image

import torch
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration, AutoTokenizer
import os, types
import subprocess

model_id = "llava-hf/llava-onevision-qwen2-0.5b-ov-hf"
# model_id = "lmms-lab/llava-onevision-qwen2-0.5b-ov"
DEFAULT_QUESTION = "What are these?"
stop_token_ids=1
import os

INFER_BACKEND = "hf"
INFER_BACKEND = os.getenv("INFER_BACKEND", INFER_BACKEND)

img_name = "000000039769.jpg"
if not os.path.exists(img_name):
    image_file = "http://images.cocodataset.org/val2017/000000039769.jpg"
    raw_image = Image.open(requests.get(image_file, stream=True).raw)
else:
    raw_image = Image.open(img_name)
    

    if INFER_BACKEND == "hf":
        model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            model_id, 
            torch_dtype=torch.float16, 
            low_cpu_mem_usage=True, 
        ).to(0)

        processor = AutoProcessor.from_pretrained(model_id)

        # Define a chat history and use `apply_chat_template` to get correctly formatted prompt
        # Each value in "content" has to be a list of dicts with types ("text", "image") 
        conversation = [
            {

            "role": "user",
            "content": [
                {"type": "text", "text": DEFAULT_QUESTION},
                {"type": "image"},
                ],
            },
        ]
        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        print(prompt, type(prompt))

        inputs = processor(images=raw_image, text=prompt, return_tensors='pt').to(0, torch.float16)

        if True:
            # print(inputs["input_ids"].shape, processor)
            # attention_mask = torch.ones(input_ids.shape,dtype=torch.long,device=device)
            # inputs['pad_token_id'] = 0
            print(inputs.keys())
            print(inputs['input_ids'].shape, inputs['attention_mask'].shape)


        output = model.generate(**inputs, max_new_tokens=200, do_sample=False)
        print(output[0][:2], " || ", output[0].shape)
        print(processor.decode(output[0][:], skip_special_tokens=True))


    if INFER_BACKEND == "vllm":
        def get_prompt(question:str, modality: str):
            if modality == "video":
                prompt = f"<|im_start|>user <video>\n{question}<|im_end|> \
                <|im_start|>assistant\n"

            elif modality == "image":
                prompt = f"<|im_start|>user <image>\n{question}<|im_end|><|im_start|>assistant\n"
            return prompt
        
        from vllm import LLM, SamplingParams
        prompt = get_prompt(DEFAULT_QUESTION, "image")
        print(prompt)
        llm = LLM(model=model_id, max_model_len=8192)
        sampling_params = SamplingParams(temperature=0.0,
                                            max_tokens=200,
                                            stop_token_ids=None)
        inputs_one=[]
        inputs_one.append({
                        "prompt": prompt,
                        "multi_modal_data": {
                            "image": raw_image
                        },
                    })

        outputs = llm.generate(inputs_one, sampling_params=sampling_params)

        print("vllm: \n")
        for o in outputs:
            generated_text = o.outputs[0].text
            print(generated_text)
        

def main(model_id: str = "llava-hf/llava-onevision-qwen2-0.5b-ov-hf", output_dir: str = 'model_files/', action: str = None, num_layers: int = 32, test: bool = False, input = DEFAULT_QUESTION, device = None, max_tokens = 200):
    model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            model_id, 
            torch_dtype=torch.float16, 
            low_cpu_mem_usage=True, 
        )
    model.eval()
    processor = None # AutoProcessor.from_pretrained(model_id)
    
    if not device:
        device = torch.device("cuda")
        model.to(device)
    else:
        model.to(device)
        device = model.device
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Export the model based on the specified export type
    if action == 'vision_tower':
        def get_image_features(self, pixel_values: torch.Tensor):
            assert pixel_values.ndim == 4, "Input tensor must have 4 dimensions"
            vision_feature_layer = -1
            image_features = self.vision_tower(pixel_values, output_hidden_states=True)
            selected_image_feature = image_features.hidden_states[vision_feature_layer]

            image_features = self.multi_modal_projector(selected_image_feature)
            return image_features
        
        model.backup_forward = model.forward
        model.forward = types.MethodType(get_image_features, model)
        
        dynamic_axes={'input': {0: 'N'},'output': {0: 'N'}}
     
        out_path = os.path.join(output_dir, f"{action}.onnx")

        input_img = torch.zeros((5, 3, 384, 384)).to(model.device, torch.float16)
        print(f'start exporting {out_path}')
        torch.onnx.export(model,
                        args=(input_img,),
                        f=out_path,
                        opset_version=17,
                        input_names=['input'],
                        output_names=['output'] ,
                        dynamic_axes=dynamic_axes)
        # use subprocess and onnxsim:
        assert 0 == subprocess.call(["onnxsim", out_path, out_path])
        print(f'{action}: {out_path} saved.')
        model.forward = model.backup_forward
        
    elif action == 'image_newline':
        image_newline = model.image_newline.requires_grad_(False).data.cpu()
        out_path = os.path.join(output_dir, f"{action}.pt")
        torch.save(image_newline, out_path)
        print(f'{action}: {out_path} saved. Shape/Type: {image_newline.shape}/{image_newline.dtype}')
    elif action == "embed_tokens":
        embed_tokens = model.language_model.model.embed_tokens.weight.requires_grad_(False).data.cpu()
        out_path = os.path.join(output_dir, f"{action}.pt")
        torch.save(embed_tokens, out_path)
        print(f'{action}: {out_path} saved. Shape/Type: {embed_tokens.shape}/{embed_tokens.dtype}')
    elif action == "tokenizer":
        if processor is None:
            processor = AutoProcessor.from_pretrained(model_id)
        processor.save_pretrained(output_dir)
        print(f'{action}: {output_dir} saved.')
    elif action == "batchable":
        import sys
        sys.path.insert(0, "models")
        from exporting_general import export_batchable_part
        export_batchable_part(model.language_model, output_dir, num_layers=-1, use_index_select=True)
    elif action == "batchless_prefill": 
        import sys
        sys.path.insert(0, "models")
        from exporting_general import export_batchless_prefill_part
        export_batchless_prefill_part(model.language_model, output_dir)
    else:
        raise ValueError(f"Unknown action: {action}")
        
if __name__ == '__main__':
    import fire
    fire.Fire(main)