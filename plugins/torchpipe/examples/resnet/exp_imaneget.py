import os
import fire
import hami
import torchpipe
from pathlib import Path
import tempfile
import torchpipe.utils.model_helper as helper

def onnx2trt(onnx_path, toml_path):
    """Convert ONNX model to TensorRT using HAMI configurations."""
    config = hami.parser.parse(toml_path)
    
    trt_path = Path(onnx_path).with_suffix('.trt')
    for _, settings in config.items():
        if 'model' in settings:
            settings['model'] = onnx_path
            settings['model::cache'] = str(trt_path)
            break
    
    kwargs = hami.Dict()
    kwargs['config'] = config
    return hami.create('Interpreter').init({}, kwargs)

def test_throughput(hami_backend):
    """Benchmark model inference performance."""
    bench = hami.init("Benchmark", {
        "num_clients": "12",
        "total_number": "20000"
    })
    
    dataset = helper.TestImageDataset()
    _, image_bytes = next(iter(dataset))
    
    bench.forward([{'data': image_bytes}]*100, hami_backend)
    result = hami.default_queue().get(block=False)
    print("Benchmark result:", result)

def test(model='resnet50'):
    """Main test pipeline: export ONNX, convert to TRT, and benchmark. `model` should be a valid timm model name."""
    onnx_path = Path(tempfile.gettempdir()) / f"{model}.onnx"
    
    if not onnx_path.exists():
        model, preprocessor = helper.get_timm_and_export_onnx(model, str(onnx_path))
    
    hami_backend = onnx2trt(str(onnx_path), f'{model}.toml')
    check_accuracy(hami_backend)

 
    
import torchpipe.utils.model_helper as helper
def dataset():
    
    ms_val_dataset = helper.get_mini_imagenet().to_torch_dataset()
    # import pdb; pdb.set_trace()
    helper.import_or_install_package('tqdm')
    from tqdm import tqdm

    for item in tqdm(ms_val_dataset, desc="Processing", position=0, leave=True):
        import pdb; pdb.set_trace()
        request_id = item['image:FILE']
        # IoC[ReadFile, Send[src_queue, max=20],SharedRequestState, ThreadPoolExecutor[src_queue,max_workers=10];ReadFile]
        # 
        category = item['category']
        infer_cls, infer_score = tester(image_file)

        true_labels.append(category)
        pred_labels.append(infer_cls)
        
        
if __name__ == "__main__":
    data = hami.init("IoC[ReadFile, Send2Queue[src_queue, max=20];ReadFile]")
    
    run = hami.init("IoC[SharedRequestState,ThreadPoolExecutor[src_queue,max_workers=10], XX; DI[ThreadPoolExecutor, XX]]")
    # ： DI[ThreadPoolExecutor, XX]] =》 DI::args DI(ThreadPoolExecutor, XX)
    fire.Fire(dataset)