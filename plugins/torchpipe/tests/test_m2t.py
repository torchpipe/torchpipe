def test_mat2tensor():
    import omniback
    import torchpipe, torch
    omniback.init("DebugLogger")
    x = omniback.init("S[Tensor2Mat,Mat2Tensor]")
    x_i = {'data':torch.zeros((224,224,3))}
    x(x_i)


if __name__ == "__main__":
    test_mat2tensor()