import pytest
import omniback as om


# from torchpipe import _C as tc
import torch
# def test_backend_creation():
#     # Test Identity backend
#     backend = _C.create("Identity")
#     assert backend is not None
#     assert backend.max() == 1

# def test_backend_initialization():
#     backend = _C.create("Identity")
#     # Test chained initialization
#     backend.init({"param1": "value1"}).init({})
    
#     # Test with empty config
#     backend.init({})

# def test_backend_execution():
#     backend = _C.create("Identity")
#     backend.init({})
    
#     # Test with different input types
#     test_cases = [
#         {"data": "string_input"},
#         {"data": 42},
#         {"data": 3.14},
#         {"data": [1, 2, 3]},
#         {"data": {"nested": "data"}}
#     ]
    
#     for test_input in test_cases:
#         input_dict = _C.Dict(test_input)
#         backend(input_dict)
#         assert pytest.approx(input_dict["result"]) == test_input["data"]

def test_backend():
    import torchpipe
    backend = om.init("With[StreamPool,ResizeTensor]", {"resize_h": "112","resize_w": "113"})
    with pytest.raises(RuntimeError):
        backend.init({})
    
    input={"data":torch.zeros((1,3,224,224))}
    backend(input)
    assert input["result"].shape == (1,3,112,113)
    
    input={"data":torch.zeros((224,224, 3))}
    backend(input)
    assert input["result"].shape == (112,113,3)
    
    input={"data":torch.zeros((224,224, 3), dtype=torch.uint8, device="cuda")}
    backend(input)
    assert input["result"].shape == (112,113,3)
    
if __name__ == "__main__":
    
    import time
    # time.sleep(4)
    
    print("import torchpipe")
    import torchpipe
    
    test_backend()