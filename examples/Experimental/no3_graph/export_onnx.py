


from  torchpipe.utils.models import onnx_export



def export_onnx_resnet18():
    import torchvision
    model = torchvision.models.resnet18(pretrained=True)
    onnx_export(model, "resnet18.onnx", None, 17)
    import onnxsim
    for i in range(3):
        model_simp, check = onnxsim.simplify("resnet18.onnx")
        assert check, "Simplified ONNX model could not be validated"
        onnx.save(model_simp, "resnet18.onnx")

    # trtexec --onnx=resnet18.onnx --fp16  --minShapes=input:1x3x224x224 --optShapes=input:16x3x224x224 --maxShapes=input:16x3x224x224 --shapes=input:16x3x224x224  --saveEngine=resnet18_16.trt
    #  18.8828 0.2
    # trtexec --onnx=resnet18.onnx --fp16  --minShapes=input:1x3x224x224 --optShapes=input:32x3x224x224 --maxShapes=input:32x3x224x224 --shapes=input:32x3x224x224  --saveEngine=resnet18_32.trt
    #  18.8828 0.2
    # trtexec --onnx=resnet18.onnx --fp16  --minShapes=input:1x3x224x224 --optShapes=input:8x3x224x224 --maxShapes=input:8x3x224x224 --shapes=input:8x3x224x224  --saveEngine=resnet18_8.trt
    #  18.8828 0.2
    # trtexec --onnx=resnet18.onnx --fp16  --minShapes=input:1x3x224x224 --optShapes=input:4x3x224x224 --maxShapes=input:4x3x224x224 --shapes=input:4x3x224x224  --saveEngine=resnet18_4.trt
    #  18.8828 0.2
    # trtexec --onnx=resnet18.onnx --fp16  --minShapes=input:1x3x224x224 --optShapes=input:1x3x224x224 --maxShapes=input:


if __name__ == '__main__':

    import fastervit, onnx
    model = fastervit.create_model("faster_vit_0_224", resolution=224, pretrained=True, exportable=True)
    onnx_export(model, "fastervit_0_224_224.onnx", None, 17)
    import onnxsim
    for i in range(3):
        model_simp, check = onnxsim.simplify("fastervit_0_224_224.onnx")
        assert check, "Simplified ONNX model could not be validated"
        onnx.save(model_simp, "fastervit_0_224_224.onnx")

    export_onnx_resnet18()
    # trtexec --onnx=fastervit_0_224_224.onnx --fp16  --minShapes=input:1x3x224x224 --optShapes=input:16x3x224x224 --maxShapes=input:16x3x224x224 --shapes=input:16x3x224x224  --saveEngine=fastervit_0_224_224_16.trt
    # 70.6406 3.29885

    # trtexec --onnx=fastervit_0_224_224.onnx --fp16  --minShapes=input:1x3x224x224 --optShapes=input:32x3x224x224 --maxShapes=input:32x3x224x224 --shapes=input:32x3x224x224  --saveEngine=fastervit_0_224_224_32.trt
    #  48.8933 5.98894 


    # trtexec --onnx=fastervit_0_224_224.onnx --fp16  --minShapes=input:1x3x224x224 --optShapes=input:8x3x224x224 --maxShapes=input:8x3x224x224 --shapes=input:8x3x224x224  --saveEngine=fastervit_0_224_224_8.trt
    # 122.463 2.1 ms 
    # trtexec --loadEngine=fastervit_0_224_224_8.trt --shapes=input:1x3x224x224 --fp16 
    # 125.014 1.6ms
    # trtexec --loadEngine=fastervit_0_224_224_8.trt --shapes=input:4x3x224x224 --fp16 
    # 133.82 1.8

    # trtexec --onnx=fastervit_0_224_224.onnx --fp16  --minShapes=input:1x3x224x224 --optShapes=input:4x3x224x224 --maxShapes=input:4x3x224x224 --shapes=input:4x3x224x224  --saveEngine=fastervit_0_224_224_4.trt
    # 125.308 1.525
    # trtexec --loadEngine=fastervit_0_224_224_4.trt --shapes=input:1x3x224x224 --fp16 
    # 83.873 1.42727


   # trtexec --onnx=fastervit_0_224_224.onnx --fp16  --minShapes=input:1x3x224x224 --optShapes=input:1x3x224x224 --maxShapes=input:1x3x224x224 --shapes=input:1x3x224x224  --saveEngine=fastervit_0_224_224_1.trt
    # 195.043 0.963574 
    # trtexec --loadEngine=fastervit_0_224_224_1.trt --shapes=input:1x3x224x224 --verbose 
