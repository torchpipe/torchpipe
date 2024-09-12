
import torchpipe
import torch, os, glob



plugin=torchpipe.utils.cpp_extension.load(name="plugin", sources=glob.glob("./*.cpp"),
                                   extra_include_paths=['/workspace/TensorRT-10.2.0.19/include/'],
                                   extra_ldflags=['-L/workspace/TensorRT-10.2.0.19/targets/x86_64-linux-gnu/lib/','-lnvinfer_plugin','-lnvinfer','-lipipe','-Wl,--no-as-needed'],
                                   verbose=True,
                                   is_python_module=False)
storage = torchpipe.ThreadSafeKVStorage.getInstance()
# kv.write("z", 1)
# print(kv.read("z"))
# exit(0)
# tp_model = torchpipe.Pipe("onnx/llama2.toml")


# export LD_LIBRARY_PATH=/workspace/TensorRT-10.2.0.19/lib:$LD_LIBRARY_PATH
# export PATH=/workspace/TensorRT-10.2.0.19/bin/:$PATH


 


# input_embeds_no_img = "/workspace/cur_input_embeds_no_im.pt"
# input_llm_embeds = torch.load(input_embeds_no_img)


from transformers import AutoTokenizer, AutoModelForCausalLM



tp_model  = torchpipe.Pipe("onnx/llama2.toml")
text = "San Francisco is a"
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
text = "San Francisco is a"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

inputs = tokenizer(text, return_tensors="pt")
print(inputs["input_ids"])

inputs = {'data':inputs["input_ids"][0],'request_id':'r0', 'node_name':'input', 'trt_plugin':'batchless_prefill'}
# inputs += [data]


tp_model(inputs)
print(inputs['result'].shape)
print(inputs['result'])

# next_tokens = torch.argmax(inputs['result'], dim=-1)
# print(next_tokens)


# generated_text = tokenizer.decode(inputs['result'][0], skip_special_tokens=True)
# torch.Size([5, 4096]) tensor(0.0018, device='cuda:0', dtype=torch.float16) tensor(-0.0039, device='cuda:0', dtype=torch.float16)
# async mode:
# events = [torchpipe.Event() for _ in range(len_data)]
if False:
    ev = torchpipe.Event(len_data)
    # assert(events[0] != events[1])
    for i in range(len_data):
        inputs[i]['event'] = ev



    

    # for i in range(len_data):
    #     events[i].Wait()
    while (not ev.Wait(1000)):
        pass

# assert(finished)
# events[1].Wait()


# import pdb; pdb.set_trace()
# print(kv['r0', 'result'])
    print(storage)
# q = kv.__getitem__('r0','tensor_item').as_queue()


    for i in range(len_data):
        print('***'*10)
        request_id = inputs[i]['request_id']
        q = storage[request_id,'queue'].as_queue()
        print(request_id, q.size())
        storage.erase(request_id) # kvcache has arleady been erased

        print(inputs[i].keys())
        print(inputs[i]['result'])
        
        re = q.pop_all()
        print(f'len = {len(re)} data = ', re[:5], '...', re[-5:])



# torchpipe.utils.test.test_from_raw_file(run,file_dir="../assets/", num_clients=2,
#                                         total_number=1000)
