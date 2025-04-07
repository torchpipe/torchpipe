import os
from torchpipe.serve.register import BackendEngine, register_backend_engine

class CustomBackendEngine(BackendEngine):
    def __init__(self):
        super().__init__()

    def forward_async(self, *args, **kwargs):
        # Implement the logic to handle the request asynchronously
        print(args, kwargs)
        

register_backend_engine("llama2", CustomBackendEngine())

if __name__ == '__main__':
    import fire
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    # fire.Fire(main)
    from torchpipe.serve.openai.openai_server_api import main
    main()