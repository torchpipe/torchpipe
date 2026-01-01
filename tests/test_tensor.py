import os, pytest
import omniback as om




def test_create_tensor():
    pytest.importorskip("torch")
    import torch
    a=torch.zeros((1,1)).cuda()
    
    io = om.Dict({"data":a})
    b = io["data"]
    c = torch.from_dlpack(b)
    print(b, type(b), c.shape, type(c))
    
    pipeline = om.init("AsU64Identity", {})
    with pytest.raises(TypeError):
        pipeline([io])
        print(io['result'])

    
if __name__ == "__main__":
    test_create_tensor()