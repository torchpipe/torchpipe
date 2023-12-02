


import os,torch




if __name__ == "__main__":
    from torchpipe import pipe, parse_toml, TASK_DATA_KEY, TASK_RESULT_KEY
     # config["resnet50"]["model::cache"] = f"resnet50_{}_{}.trt"
    config = {
        "backend":"SyncTensor[TensorrtTensor]",
        "instance_num": 1,
        "model":"fastervit_0_224_224_32.trt",
        "batching_timeout":6
    }
    nodes = pipe(config)
    img_g = torch.zeros((1,3,224,224),device="cuda")

    def run(img):
        for img_path, img_bytes in img:
            input = {TASK_DATA_KEY: img_g, "node_name": "jpg_decoder"}
            nodes(input)

            if TASK_RESULT_KEY not in input.keys():
                print("error : no result")
                return
            z=input[TASK_RESULT_KEY].cpu()

    from torchpipe.utils.test import test_from_raw_file

 
    num_clients = 40

    result = test_from_raw_file(run, os.path.join("../../..", "test/assets/encode_jpeg/"),num_clients=num_clients, batch_size=1,total_number=40000)
