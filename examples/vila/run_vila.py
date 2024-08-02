
import torchpipe
import cv2, torch, os, glob



# plugin=
torchpipe.utils.cpp_extension.load(name="plugin", sources=glob.glob("./*.cpp"),
                                   extra_include_paths=['/workspace/TensorRT-10.2.0.19/include/'],
                                   extra_ldflags=['-L/workspace/TensorRT-10.2.0.19/targets/x86_64-linux-gnu/lib/','-lnvinfer_plugin','-lnvinfer','-lipipe','-Wl,--no-as-needed'],
                                   verbose=True,
                                   is_python_module=False)
storage = torchpipe.ThreadSafeKVStorage.getInstance()
# kv.write("z", 1)
# print(kv.read("z"))
# exit(0)
model = torchpipe.Pipe("vila.toml")


# export LD_LIBRARY_PATH=/workspace/TensorRT-10.2.0.19/lib:$LD_LIBRARY_PATH
# export PATH=/workspace/TensorRT-10.2.0.19/bin/:$PATH


def run(inputs):
    path_img, img = inputs[0]
    input = {'data':img}
    model(input)

img_path  = 'demo_images/av.png'
img = cv2.imread(img_path)
input = torch.from_numpy(img)
input_ones = torch.ones((1000,2560)).to(torch.float16).cuda()
pt = "/workspace/VILA/inputs_embeds.pt"
if os.path.exists(pt):
    input = torch.load(pt).squeeze(0).cuda()
data = {'data':input_ones,'request_id':'r0', 'node_name':'batchful', 'trt_plugin':'batchless_prefill'}

inputs = [data]
inputs += [{'data':input,'request_id':'r1', 'node_name':'batchful', 'trt_plugin':'batchless_prefill'}]

len_data = len(inputs)
# async mode:
# events = [torchpipe.Event() for _ in range(len_data)]
ev = torchpipe.Event(len_data)
# assert(events[0] != events[1])
for i in range(len_data):
    inputs[i]['event'] = ev



model(inputs)

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



# torchpipe.utils.test.test_from_raw_file(run,file_dir="../assets/", num_clients=2,
#                                         total_number=1000)
