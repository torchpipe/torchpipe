def test_mat2tensor():
    import hami
    import torchpipe, torch
    hami.init("DebugLogger")
    x = hami.init("S[Tensor2Mat,Mat2Tensor]")
    x_i = {'data':torch.zeros((224,224,3))}
    x(x_i)


if __name__ == "__main__":
    test_mat2tensor()