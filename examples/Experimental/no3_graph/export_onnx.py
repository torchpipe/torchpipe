


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

    # trtexec --onnx=fastervit_0_224_224.onnx --fp16  --minShapes=input:1x3x224x224 --optShapes=input:8x3x224x224 --maxShapes=input:8x3x224x224 --shapes=input:8x3x224x224  --saveEngine=fastervit_0_224_224.trt
    # 122.463 2.1 ms 
    # trtexec --loadEngine=fastervit_0_224_224.trt --shapes=input:1x3x224x224 --fp16 
    # 125.014 1.6ms
    # trtexec --loadEngine=fastervit_0_224_224.trt --shapes=input:4x3x224x224 --fp16 
    # 133.82 1.8

    # trtexec --onnx=fastervit_0_224_224.onnx --fp16  --minShapes=input:1x3x224x224 --optShapes=input:4x3x224x224 --maxShapes=input:4x3x224x224 --shapes=input:4x3x224x224  --saveEngine=fastervit_0_224_224_4.trt
    # 125.308 1.525
    # trtexec --loadEngine=fastervit_0_224_224_4.trt --shapes=input:1x3x224x224 --fp16 
    # 83.873 1.42727


   # trtexec --onnx=fastervit_0_224_224.onnx --fp16  --minShapes=input:1x3x224x224 --optShapes=input:1x3x224x224 --maxShapes=input:1x3x224x224 --shapes=input:1x3x224x224  --saveEngine=fastervit_0_224_224_1.trt
    # 195.043 0.963574 
    # trtexec --loadEngine=fastervit_0_224_224_1.trt --shapes=input:1x3x224x224 --verbose 
