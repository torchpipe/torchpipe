
import torchpipe
import cv2, torch, os, glob

torchpipe.utils.cpp_extension.load(name="plugin", sources=glob.glob("./*.cpp"),
                                   extra_include_paths=['/workspace/TensorRT-10.2.0.19/include/'],
                                   extra_ldflags=['-L/workspace/TensorRT-10.2.0.19/targets/x86_64-linux-gnu/lib/','-lnvinfer_plugin','-lnvinfer'],
                                   verbose=True)

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
input = torch.ones((244,2560)).to(torch.float16).cuda()
pt = "/workspace/VILA/inputs_embeds.pt"
if os.path.exists(pt):
    input = torch.load(pt).squeeze(0).cuda()
data = {'data':input,'request_id':'zzz', 'node_name':'batchful', 'trt_plugin':'batchless_prefill'}
model(data)
print(data['result'].shape)
# torchpipe.utils.test.test_from_raw_file(run,file_dir="../assets/", num_clients=2,
#                                         total_number=1000)
