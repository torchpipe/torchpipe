

from omniback.utils.test import test_from_ids
import gc

import torch
torch.set_num_threads(8)

def main(
    model_name: str = "eva02_base_patch14_448.mim_in22k_ft_in22k_in1k",
    image_path: str = "../../tests/assets/encode_jpeg/grace_hopper_517x606.jpg",
    num_clients: int = 1,
    total_number: int = 5000,
    target:list = ['torchpipe', 'torchpipe-thrift', 'torch2trt', 'triton']
):
    """
    Convert and run inference on timm model using torch2trt.
    """
    results = {}
    ids = list(range(total_number))
    
    target = ['torchpipe', 'torchpipe-thrift']
    if num_clients == 1:
        target += ['torch2trt']
    
    if 'torch2trt' in target:
        assert torch.__version__ < "2.9", "pls exporting onnx with torch<2.9 (or set dynamo=False)"
        from run_torch2trt import get_client
        client = get_client(image_path)
        results['torch2trt'] = test_from_ids([client]*num_clients, ids)
        del client
        gc.collect()
    
    # apt install -y thrift-compiler
    # cd benchmarks && thrift --gen py server.thrift && mv gen-py/* ./
    # pip install thrift
    if 'torchpipe-thrift' in target:
        from run_torchpipe import get_thrift_client
        forward_funcs, clients = get_thrift_client(
            image_path, model_path=model_name+".onnx", num_clients=num_clients, port=1020)
        results['torchpipe-thrift'] = test_from_ids(
            forward_funcs, ids)
        del clients
        gc.collect()
        
    # export onnx
    if 'torchpipe' in target:
        from run_torchpipe import get_client
        client = get_client(image_path, model_path=model_name+".onnx")
        results['torchpipe'] = test_from_ids([client]*num_clients, ids)
        del client
        gc.collect()


    


    
    if 'triton' in target:
        assert False
        from triton_client import TritonClient
        client = TritonClient(model_name=model_name)
        results['triton'] = test_from_ids([client]*num_clients, ids)
        del client
        gc.collect()

    return results






if __name__ == "__main__":
    import fire
    fire.Fire(main)