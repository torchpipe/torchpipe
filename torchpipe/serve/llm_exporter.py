import os
import torch
from transformers import LlamaForCausalLM, OPTForCausalLM, PreTrainedModel, PreTrainedTokenizerFast



def batchful_export(llm: PreTrainedModel, inputs = None, out_dir = 'onnx/'):
    class Wrapper(torch.nn.Module):
        def __init__(self, llm):
            super().__init__()
            self.llm = llm
        def forward(self, inputs_embeds):
            with torch.no_grad():
                out = self.llm.forward(input_ids = None,
                                attention_mask = None,
                                past_key_values = None,
                                inputs_embeds = inputs_embeds,
                                return_dict=False)

            assert isinstance(out[0], torch.Tensor)
            print('out: ', out[0].shape, out[0].dtype)
            # print(out[0])
            return out[0]
            # next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            # return next_tokens
    model = Wrapper(llm).eval()

    
    dynamic_axes={'inputs_embeds': {0: 'seq_len'}, 'logits': {0: 'seq_len'}}
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "batchful.onnx")

    if inputs is None:
        # inputs = torch.randint(0, 50256, (1, 244)).to(model.llm.device, torch.long)
        # embed_tokens = None
        for name, param in model.named_parameters():
            if "embed_tokens" in name:
                inputs = torch.zeros((244, param.shape[1])).to(model.llm.device, param.dtype)
                break


        
    
    torch.onnx.export(model,
                    args=(inputs,),
                    f=out_path,
                    opset_version=17,
                    input_names=['inputs_embeds'],
                    output_names=['logits'] ,
                    dynamic_axes=dynamic_axes)
    
    # export_modules_as_functions={modeling_llama.TorchPlugin}
    print(f'Batchful CausalLM: {out_path} saved.')

def get_model(name_or_path):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(name_or_path,attn_implementation='eager')
    model.eval()

    # 将模型转换为 fp16
    model.half()
    print(model)

    # 将模型移动到 GPU
    device = 'cpu' # torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model

if __name__ == "__main__":
    
    

    import torch
    from transformers.modeling_attn_mask_utils import AttentionMaskConverter

    converter = AttentionMaskConverter(True)
    print(converter.to_4d(torch.tensor([[1, 1, 1, 1, 1]]), 5, key_value_length=5, dtype=torch.float16))
                    # tensor([[[[-3.4028e+38, -3.4028e+38, -3.4028e+38, -3.4028e+38, -3.4028e+38],
                    #         [-3.4028e+38, -3.4028e+38, -3.4028e+38, -3.4028e+38, -3.4028e+38],
                    #         [-3.4028e+38, -3.4028e+38, -3.4028e+38, -3.4028e+38, -3.4028e+38],
                    #         [-3.4028e+38, -3.4028e+38, -3.4028e+38,  0.0000e+00, -3.4028e+38],
                    #         [-3.4028e+38, -3.4028e+38, -3.4028e+38,  0.0000e+00,  0.0000e+00]]]])

    exit(0)
    model = get_model("facebook/opt-125m")
    
    batchful_export(model)
    
