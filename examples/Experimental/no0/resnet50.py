import torch
import cv2
import os
import torchpipe as tp


import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--config', dest='config', type=str,  default="./resnet50.toml",
                    help='configuration file')

args = parser.parse_args()


def export_onnx(onnx_save_path):
    import torch
    import torchvision.models as models
    resnet50 = models.resnet50().eval()
    x = torch.randn(1,3,224,224)
    onnx_save_path = "./resnet50.onnx"
    tp.utils.models.onnx_export(resnet50, onnx_save_path, x)
    

def draw(show_keys, result):
    color = ["red", "green", "yellow", "blue", "black", "pink", "gray", "orange", "purple"]
    import matplotlib.pyplot as plt 

    datas = []
    keys = show_keys.keys()
    for key in keys:
        data = []
        for k,v in result.items():
            data.append(float(v[key]))
        datas.append(data)

    num_clients = [x for x in result.keys()]

    from matplotlib.backends.backend_pdf import PdfPages
    pp = PdfPages('resnet.pdf')

    plt.rcParams.update({'font.size': 20})
    fig,ax=plt.subplots(len(keys),1,figsize=(10,8))
    
    for i in range(len(keys)):
        ax[i].bar(num_clients, datas[i],color=color[i])
        ax[i].set_xticks(num_clients)
        
        ax[i].legend([show_keys[keys[i]]])
    ax[len(keys)-1].set_xlabel('Number of Clients')

    plt.savefig('resnet.svg')
    pp.savefig(plt.gcf())

    # plt.savefig('resnet.png')

    #%%
    plt.show()
    pp.close()

    print("done. Saved to resnet.pdf and resnet.svg")
 
if __name__ == "__main__":


    onnx_save_path = "./resnet50.onnx"
    if not os.path.exists(onnx_save_path):
        export_onnx(onnx_save_path)

    img_path = "../../test/assets/image/gray.jpg"
    img=open(img_path,'rb').read()

    toml_path = args.config 
    
    from torchpipe import pipe, parse_toml, TASK_DATA_KEY, TASK_RESULT_KEY
    config = parse_toml(toml_path)
    # config["resnet50"]["model::cache"] = f"resnet50_{}_{}.trt"
    nodes = pipe(config)

    def run(img):
        for img_path, img_bytes in img:
            input = {TASK_DATA_KEY: img_bytes, "node_name": "jpg_decoder"}
            nodes(input)

            if TASK_RESULT_KEY not in input.keys():
                print("error : no result")
                return
            z=input[TASK_RESULT_KEY].cpu()


    run([(img_path, img)])

    from torchpipe.utils.test import test_from_raw_file

    results = {}
    for i in range(1,17):
        num_clients = i
        if num_clients%2 == 1:
            continue
        result = test_from_raw_file(run, os.path.join("../..", "test/assets/encode_jpeg/"),num_clients=num_clients, batch_size=1,total_number=10000)
        results[num_clients]=result

    print(results)

    show_keys= {"throughput::qps":"QPS", 
               "latency::TP50":"TP50",
               "gpu_usage":"GPU Usage",}


    draw(show_keys, results)