import torch
import tvm_ffi

def create_new_stream_if_using_default():
    if torch.cuda.current_stream() != torch.cuda.default_stream():
        return False

    context = tvm_ffi.use_torch_stream(torch.cuda.stream(torch.cuda.Stream()))
    context.__enter__()
    return True

def get_current_cuda_stream():
    return torch.cuda.current_stream().cuda_stream

def wait_for_default_stream():
    if torch.cuda.current_stream() != torch.cuda.default_stream():
        torch.cuda.current_stream().wait_stream(torch.cuda.default_stream())

def current_stream_synchronize():
    torch.cuda.current_stream().synchronize()
