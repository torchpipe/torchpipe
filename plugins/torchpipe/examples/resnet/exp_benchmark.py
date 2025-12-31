import os
import fire
import omniback
import torchpipe
from pathlib import Path
import tempfile
import torchpipe.utils.model_helper as helper

def onnx2trt(onnx_path, toml_path):
    """Convert ONNX model to TensorRT using OMNI configurations."""
    config = omniback.parser.parse(toml_path)
    
    # trt_path = Path(onnx_path).with_suffix('.trt')
    trt_path = onnx_path.replace('.onnx','_benchmark.trt')
    for _, settings in config.items():
        if 'model' in settings:
            settings['model'] = onnx_path
            settings['model::cache'] = str(trt_path)
            break
    
    kwargs = omniback.Dict()
    kwargs['config'] = config
    return omniback.create('Interpreter').init({}, kwargs)

def test_throughput(omniback_backend):
    """Benchmark model inference performance."""
    bench = omniback.init("Benchmark", {
        "num_clients": "12",
        "total_number": "20000"
    })
    
    dataset = helper.TestImageDataset()
    _, image_bytes = next(iter(dataset))
    
    bench.forward_with_dep([omniback.Dict({'data': image_bytes})]*100, omniback_backend)
    result = omniback.default_queue().get(True)
    print(type(result))
    print("Benchmark result:", result)

def test(model='resnet50'):
    """Main test pipeline: export ONNX, convert to TRT, and benchmark. `model` should be a valid timm model name."""
    onnx_path = Path(tempfile.gettempdir()) / f"{model}.onnx"
    
    if not onnx_path.exists():
        helper.get_timm_and_export_onnx(model, str(onnx_path))
    
    omniback_backend = onnx2trt(str(onnx_path), f'{model}.toml')
    test_throughput(omniback_backend)

if __name__ == "__main__":
    fire.Fire(test)