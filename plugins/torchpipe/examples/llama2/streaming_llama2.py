import os
from torchpipe.serve.register import BackendEngine, register_engine
from torchpipe.serve.openai.openai_server_api import SamplingParams
from torchpipe.serve.output import RequestOutput, SequenceOutput,  Status, StatusCode
import shortuuid
import hami
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from typing import List




class CustomBackendEngine(BackendEngine):
    def __init__(self, *args, **kwargs):
        super().__init__()
        
        from plain_llama2 import PyPlugin
        hami.register("TorchPlugin", PyPlugin)
        hami.register("custom_backend_engine", self)
        
        exported_params = "./exported_params"
        self.tokenizer = AutoTokenizer.from_pretrained(exported_params)
        self.model = hami.init_from_file('config/streaming_llama2.toml')
        self.request_state = {}
        
        self.contiguous_batching  = hami.get("node.contiguous_batching")
       
        

    async def forward_async(self, *args, **kwargs):
        # Implement the logic to handle the request asynchronously
        print('CustomBackendEngine: ', args, kwargs)
        request_callback = kwargs.pop("callback")
        request_id = f"cmpl-{shortuuid.random()}"
        prompt = kwargs.pop("prompt")
        sampling_params: SamplingParams = kwargs.pop("sampling_params") 
        
        inputs = self.tokenizer(prompt, return_tensors="pt", return_attention_mask=False)
        input_ids = inputs['input_ids'][0]
        print(f'input_ids ', type(input_ids))
        event = hami.Event()
        io = hami.Dict({
            'data': input_ids,
            'node_name': 'embed_token',
            'event': event,
        })
        io[hami.TASK_REQUEST_ID_KEY] = request_id
        io[hami.TASK_MSG_KEY] = hami.TypedDict({"req_tokens": len(input_ids),
                                                "max_tokens": sampling_params.max_tokens,
                                                "context_length":4096})
        print("io ", io)
        
        self.request_state[request_id] = (event, request_callback, sampling_params)
        
        event.set_final_callback(lambda : self.final_callback(request_id))
        self.model(io)
    
        
    def final_callback(self, request_id):
        print('final_callback in')
        local_ev, request_callback, _ = self.request_state[request_id]
        
        output = RequestOutput()
        
        try:
            local_ev.try_throw()
        except Exception as e:
            print('exception')
            
            output.status = Status(StatusCode.UNKNOWN, str(e))
            output.usage = None
            output.finished = True
            print("ERROR MSG: \n", e)
            self.finish_batching(request_id)
            request_callback(output)
        else:
            print('no exception')
            
        del self.request_state[request_id]
        self.finish_batching(request_id)
        
    def forward(self, io: List[hami.Dict]):
        request_id = io[0]['request_id']#.decode('utf-8')
        data = io[0]['data']
        text = self.tokenizer.decode(data, skip_special_tokens=True)
        
        print(io, 'forward', request_id)
        
        _, request_callback, sampling_params = self.request_state[request_id]
        
        seq = SequenceOutput()
        output = RequestOutput()
        output.status = Status(StatusCode.OK, "OK")
        output.usage = None
        
        if text in sampling_params.stop_token_ids:
            seq.finish_reason = "stop"
            output.finished = True
            print("STOP TOKEN ID: \n", text)
        else:
            if "finish_reason" in io[0]:
                seq.finish_reason = io[0]["finish_reason"]
                output.finished = True
                io[0].remove("restart")
            else:
                output.finished = False 
            seq.text = text
            
        output.outputs = [seq]
        
        if output.finished:
            print(f"finish: {request_id}")
            self.finish_batching(request_id)
            print("finished")
            
        request_callback(output)
        io[0]['result'] = io[0]['data']
        
    def finish_batching(self, req_id):
        io = hami.Dict({})
        io[hami.TASK_REQUEST_ID_KEY] = req_id
        io[hami.TASK_MSG_KEY] = hami.TypedDict({"finish": True})
        ev = io.set_event()
        ev.set_exception_callback(lambda e: self.python_callback(e)) # would not happen
        self.contiguous_batching(io)

    def python_callback(self, e):
        if isinstance(e, RuntimeError):
            print(f"Fatal RuntimeError: {e}")
        else:
            print(f"Fatal unknown exception: {e}")
            
            
register_engine("llama2", CustomBackendEngine())

if __name__ == '__main__':
    import fire
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    # fire.Fire(main)
    import sys
    sys.argv = [
        "openai_server_api.py", 
        "--model_id", "my_model",
        "--model", "llama2",
        "--host", "0.0.0.0",
        "--port", "8000",
        "--log_level", "info",
    ]
    from torchpipe.serve.openai.openai_server_api import main
    main()