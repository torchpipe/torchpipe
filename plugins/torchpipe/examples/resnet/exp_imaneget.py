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

 

def onnx2trt(onnx_path, toml_path, register_name):
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
    return hami.create('Interpreter', register_name).init({}, kwargs)

import torchpipe.utils.model_helper as helper
def dataset():
    
    ms_val_dataset = helper.get_mini_imagenet().to_torch_dataset()
    # import pdb; pdb.set_trace()
    helper.import_or_install_package('tqdm')
    from tqdm import tqdm

    for item in tqdm(ms_val_dataset, desc="Processing", position=0, leave=True):
        import pdb; pdb.set_trace()
        request_id = item['image:FILE']
        # IoCV0[ReadFile, Send[src_queue, max=20],SharedRequestState, ThreadPoolExecutor[src_queue,max_workers=10];ReadFile]
        # 
        category = item['category']
        infer_cls, infer_score = tester(image_file)

        true_labels.append(category)
        pred_labels.append(infer_cls)
        
        
if __name__ == "__main__":
    import time
    time.sleep(10)
    
    # queue_backend = hami.init("")
    # data = {}
    # queue_backend(data)
    # queue = data['result']
    
    # 
    
    data_pipeline = hami.init("S[ReadFile, Send2Queue(src_queue, max=10)]")
    
    model='resnet50'
    onnx_path = Path(tempfile.gettempdir()) / f"{model}.onnx"
    if not onnx_path.exists():
        helper.get_timm_and_export_onnx(model, str(onnx_path))
    resnet50 = onnx2trt(str(onnx_path), f'{model}.toml', 'trt_model')
    # pool = hami.init("IoC[Profile,ThreadPoolExecutor(out=thrd_pool,max_workers=10); DI[ThreadPoolExecutor,Profile,trt_model]]", register_name='pool')  # target_queu(default)
    pool = hami.init("IoC[Profile,ThreadPoolExecutor(out=thrd_pool,max_workers=10), Identity; DI[ThreadPoolExecutor,Identity]]", register_name='pool')  # target_queu(default)

    q = hami.default_queue(tag = 'src_queue')
    pool({'data':q}) # async
    
    
    from importlib.resources import files
    # 获取 retina.jpg 的路径
    retina_path = str(files("skimage.data").joinpath("retina.jpg"))
    import cv2
    img = cv2.imread(retina_path)
    img = cv2.resize(img, (224,224))
    retina_path=retina_path.replace(".jpg", '1.jpg')
    cv2.imwrite(retina_path, img)
    # 打印路径
    print(retina_path)
    retina_path= str(retina_path)

    thrd_pool = hami.default_queue('thrd_pool')
    index = 0
    while (q.status() == hami.Queue.RUNNING and index< 10000):
        data_pipeline({'data': (retina_path), 'request_id': (retina_path),'node_name':'dummy'})
        index += 1
        if (index % 100 == 0):
            print(index, q.size(), thrd_pool.size())
    q.join()
    
    
    # print(thrd_pool.size())
    # result = thrd_pool.get()[0]['result']
    # print(thrd_pool.size(),  hami.default_queue().size())
    # # import pdb;pdb.set_trace()
    # profile_result = hami.default_queue().get()
    # print(profile_result)
    
    # fire.Fire(dataset)
    
    
    