import os
from torchpipe.serve.register import BackendEngine, register_engine
from torchpipe.serve.openai.openai_server_api import SamplingParams
from torchpipe.serve.output import RequestOutput, SequenceOutput,  Status, StatusCode
import shortuuid
import hami
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from typing import List, Callable, Any
from dataclasses import dataclass, astuple, field
import torch


@dataclass
class RequestState:
    # 没有默认值的字段必须放在前面
    event: Any  # 替换为实际类型，如 asyncio.Event 或 hami.Event
    request_callback: Callable
    sampling_params: Any  # 替换为 SamplingParams
    args: Any
    kwargs: Any
    
    # 有默认值的字段放在后面，用 field 处理可变类型
    req_tokens: List = field(default_factory=list)
    new_tokens: List = field(default_factory=list)
    pause: bool = False
    
    def astuple(self):
        return astuple(self)

# class OnPause:
#     def forward(self, ios: List[hami.Dict]):
#         ids = []
#         for io in ios:
#             # id = io['request_id']
#             ids.append(io['request_id'])
#         print(f"paused: {ids}")
            
# class OnResume:
#     def forward(self, ios: List[hami.Dict]):
#         pass
    
class CustomBackendEngine(BackendEngine):
    def __init__(self, *args, **kwargs):
        super().__init__()
        
        from plain_llama2 import PyPlugin, get_page_table, page_size
        hami.register("TorchPlugin", PyPlugin)
        hami.register("custom_backend_engine", self)
        self.page_table = get_page_table()
        self.page_size = page_size
        exported_params = "./exported_params"
        self.tokenizer = AutoTokenizer.from_pretrained(exported_params)
        self.model = hami.init_from_file('config/streaming_llama2.toml')
        self.request_status = {}
        
        self.contiguous_batching  = hami.get("node.contiguous_batching")
        
       
        

    async def forward_async(self, *args, **kwargs):
        # Implement the logic to handle the request asynchronously
        print('CustomBackendEngine: ', args, kwargs)
        request_callback = kwargs.pop("callback")
        request_id = f"cmpl-{shortuuid.random()}"
        
        input_ids = kwargs.pop("input_ids", None)
        if input_ids is None:
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
        
        self.request_status[request_id] = RequestState(event, request_callback, sampling_params, args, kwargs, input_ids, [])

        event.set_final_callback(lambda : self.final_callback(request_id))
        self.model(io)
    

        
    def final_callback(self, request_id):
        print('final_callback in')
        status = self.request_status.pop(request_id)
        
        output = RequestOutput()
        
        try:
            status.event.try_throw()
        except Exception as e:
            print('into exception handle')
            
            output.status = Status(StatusCode.UNKNOWN, str(e))
            output.usage = None
            output.finished = True
            print("ERROR MSG: \n", e)
            # no need
            # self.finish_contiguous_batching(request_id)
            status.request_callback(output)
        else:
            print('no exception')

            self.contiguous_batching_action(request_id)
            if status.pause:
                status.kwargs["input_ids"] = status.req_tokens + status.new_tokens
                if status.sampling_params.max_tokens != 0:
                    status.sampling_params.max_tokens -= len(status.req_tokens)
                status.kwargs['sampling_params'] = status.sampling_params
                self.forward_async(*status.args, **status.kwargs)
                print("status.pause")
        
        # del self.request_status[request_id]
        print("finish final callback")
        
        
    def forward(self, ios: List[hami.Dict]):
        
        for io in ios:
            request_id = io['request_id']#.decode('utf-8')
            data = io['data']#.cpu()
            text = self.tokenizer.decode(data, skip_special_tokens=True)
            
            print(io, 'forward', request_id)
            
            # _, request_callback, sampling_params 
            status = self.request_status[request_id]
            status.new_tokens.append(data)
            
            seq = SequenceOutput()
            output = RequestOutput()
            output.status = Status(StatusCode.OK, "OK")
            output.usage = None
            
            finish_reason = None
            
            if text in status.sampling_params.stop_token_ids:
                seq.finish_reason = "stop"
                output.finished = True
                finish_reason = seq.finish_reason
                print("STOP TOKEN ID: \n", text)
            elif "finish_reason" in io:
                finish_reason = io["finish_reason"]
                if finish_reason == "length":
                    seq.finish_reason = io["finish_reason"]
                    output.finished = True
                elif finish_reason == "no_page":
                    output.finished = False 
                    status.pause = True
                else:
                    raise RuntimeError("unsupported reason: "+finish_reason)
                print(f"finish because {seq.finish_reason}")
                assert not io.contains("restart")  
                seq.text = text
            else:
                output.finished = False 
                seq.text = text
                
            output.outputs = [seq]
            
            if finish_reason:
                io['finish_reason'] = finish_reason
            status.request_callback(output)
            
            io['result'] = io['data']
        
        

                
        
    
    # def waiting_event(self, req_id, ev):
    #     io = hami.Dict({})
    #     io[hami.TASK_REQUEST_ID_KEY] = req_id
    #     io[hami.TASK_WAITING_EVENT_KEY] = ev
    #     self.contiguous_batching(io)
    
    def contiguous_batching_action(self, req_id): # finish pause2prefill pause decode2prefill
        io = hami.Dict({})
        io[hami.TASK_REQUEST_ID_KEY] = req_id
        io[hami.TASK_MSG_KEY] = hami.TypedDict({"action": "finish"})
        ev = io.set_event()
        ev.set_exception_callback(lambda e: self.python_callback(e)) # would not happen
        self.contiguous_batching(io)

    # def finish_contiguous_batching(self, req_id):
    #     io = hami.Dict({})
    #     io[hami.TASK_REQUEST_ID_KEY] = req_id
    #     io[hami.TASK_MSG_KEY] = hami.TypedDict({"action": "finish"})
    #     ev = io.set_event()
    #     ev.set_exception_callback(lambda e: self.python_callback(e)) # would not happen
    #     self.contiguous_batching(io)

    def python_callback(self, e):
        # if isinstance(e, RuntimeError):
        #     print(f"Fatal RuntimeError: {e}")
        # else:
        #     print(f"Fatal unknown exception: {e}")
        raise e
    def free_pages(self):
        pass
            
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