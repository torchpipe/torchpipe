
from image_processing import ImageProcessingService
from torchpipe import pipe

class Handler:
    def __init__(self, model_path):
        self.pipe_instance = pipe({
            "preprocessor": {
                "backend": "S[DecodeTensor,ResizeTensor,CvtColorTensor,SyncTensor]", 
                # "backend": "S[DecodeMat,ResizeMat,CvtColorMat,Mat2Tensor,SyncTensor]",
                'instance_num': 2,
                'color': 'rgb',
                'resize_h': '448',
                'resize_w': '448',
                'next': 'model',

            },
            "model": {
                "backend": "S[TensorrtTensor,CpuTensor,SyncTensor]",
                "model": model_path,
                "model::cache": model_path.replace(".onnx", ".trt"),
                "max": '4',
                'batching_timeout': 4,  # ms, timeout for batching
                'instance_num': 2,
                'mean': "123.675, 116.28, 103.53",
                'std': "58.395, 57.120, 57.375",  # merged into trt
            }}
        )

    def process_image(self, request):
        data = request.image_data
        ldata = {'data': data}
        self.pipe_instance(ldata)
        result = ldata['result']
        response = ImageProcessingService.ProcessResponse(
            success=True, result_data="123.675, 116.28, 103.53", error_message=None)
        return response


def start_thrift_server(model_path='eva02_base_patch14_448.mim_in22k_ft_in22k_in1k.onnx', port=3303):
    # import sys, os
    # sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'gen-py'))
    from thrift.protocol import TBinaryProtocol
    from thrift.transport import TTransport
    from thrift.transport import TSocket  # noqa
    from thrift.server import TServer

    handler = Handler(model_path)
    processor = ImageProcessingService.Processor(handler)
    transport = TSocket.TServerSocket(host='127.0.0.1', port=port)
    tfactory = TTransport.TBufferedTransportFactory()
    pfactory = TBinaryProtocol.TBinaryProtocolFactory()

    server = TServer.TThreadPoolServer(
        processor, transport, tfactory, pfactory)
    server.setNumThreads(10)
    print(f'Server started. port={port}')
    server.serve()
    print('done.')
    
if __name__ == "__main__":
    import fire
    fire.Fire(start_thrift_server)
