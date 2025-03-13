import hami
import torchpipe

import os
# interp = hami.init("Interpreter", {"backend": "Identity"})


def onnx2trt(onnx_path, toml_path):
    config = hami.parser.parse(toml_path)
    for k, v in config.items():
        if 'model' in v.keys():
            v['model'] = onnx_path
        v['model::cache'] = onnx_path.replace(".onnx",'.trt')

    dict_config = hami.Dict()
    dict_config['config'] = config
    pipe = hami.create('Interpreter').init({}, dict_config)
    print("config = ",config)
    return pipe

        
    # def __call__(self, x):
    #     data = {'data': x}
    #     self.model(data)
    #     return data['result']
import torchpipe.utils.model_helper as helper
def dataset():
    
    ms_val_dataset = helper.get_mini_imagenet()
    helper.import_or_install_package('tqdm')
    from tqdm import tqdm

    tester.model.cuda()
    for item in tqdm(ms_val_dataset, desc="Processing", position=0, leave=True):
        image_file = item['image:FILE']
        category = item['category']
        infer_cls, infer_score = tester(image_file)

        true_labels.append(category)
        pred_labels.append(infer_cls)


def test_throughput(model):
    bench = hami.init("Benchmark", {"num_clients": "12", "total_number": "10000"})

    dataset = helper.TestImageDataset()
    # raise 1
    for image_id, image_bytes in dataset:
        
        bench.forward([{'data': image_bytes}]*100, model)

        break
    result = hami._C.default_output_queue().get()
    print(image_id, result)
# def test_throughput(dependency):
#     bench = hami.init("Benchmark", {"num_clients": "4", "total_number": "10000"})
#     bench.forward([data]*100, dependency)
if __name__ == "__main__":
    toml_path = 'resnet50.toml'
    import tempfile
    onnx_path = os.path.join(tempfile.gettempdir(), "resnet50.onnx")

    model = onnx2trt(onnx_path, toml_path)

    test_throughput(model)