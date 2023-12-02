


from  torchpipe.utils.models import onnx_export





if __name__ == '__main__':

    import fastervit, onnx
    model = fastervit.create_model("faster_vit_0_224", resolution=224, pretrained=None, exportable=True)
    onnx_export(model, "fastervit_0_224_224.onnx", None, 17)
    import onnxsim
    for i in range(3):
        model_simp, check = onnxsim.simplify("fastervit_0_224_224.onnx")
        assert check, "Simplified ONNX model could not be validated"
        onnx.save(model_simp, "fastervit_0_224_224.onnx")