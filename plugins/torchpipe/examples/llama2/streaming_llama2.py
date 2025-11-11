import os
from torchpipe.serve.register import BackendEngine, register_engine
from torchpipe.serve.openai.openai_server_api import SamplingParams
from torchpipe.serve.output import RequestOutput, SequenceOutput,  Status, StatusCode
import shortuuid
import omniback
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from typing import List, Callable, Any
from dataclasses import dataclass, astuple, field
import torch
from tokenizers.decoders import ByteFallback, DecodeStream
import tokenizers

CURRENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))

@dataclass
class RequestState:
    # 没有默认值的字段必须放在前面
    event: Any  # 替换为实际类型，如 asyncio.Event 或 omniback.Event
    request_callback: Callable
    sampling_params: Any  # 替换为 SamplingParams
    args: Any
    kwargs: Any
    
    # 有默认值的字段放在后面，用 field 处理可变类型
    req_tokens: List = field(default_factory=list)
    new_tokens: List = field(default_factory=list)
    dropped: bool = False
    decode_stream: DecodeStream = field(default_factory=lambda: DecodeStream(skip_special_tokens=True))
    time: float = 0.0
    
    def astuple(self):
        return astuple(self)

# class OnPause:
#     def forward(self, ios: List[omniback.Dict]):
#         ids = []
#         for io in ios:
#             # id = io['request_id']
#             ids.append(io['request_id'])
#         print(f"paused: {ids}")
            
# class OnResume:
#     def forward(self, ios: List[omniback.Dict]):
#         pass
    
class CustomBackendEngine(BackendEngine):
    def __init__(self, *args, **kwargs):
        super().__init__()
        
        # from plain_llama2 import PyPlugin, page_size, get_num_layers
        use_trt = kwargs.get("use_trt", False)
        if use_trt:
            from plain_llama2_trt import PyPlugin, page_size, get_num_layers, clean_up
        else:
            from plain_llama2 import PyPlugin, page_size, get_num_layers, clean_up
            
        omniback.register("TorchPlugin", PyPlugin)
        omniback.register("custom_backend_engine", self)
        self.page_table =  omniback.default_page_table()
        self.page_size = page_size
        exported_params = "./exported_params"
        assert os.path.exists(exported_params), f"dir exported_params not found: {exported_params}"
        self.old_tokenizer = AutoTokenizer.from_pretrained(exported_params, use_fast=True)
        self.tokenizer=tokenizers.Tokenizer.from_file(exported_params + "/tokenizer.json")
        print(self.tokenizer)
        self.eos_token_id = self.old_tokenizer.eos_token_id
        # self.eos_token_id = self.tokenizer.get_vocab()["</s>"]

        # import pdb; pdb.set_trace()
        config = os.path.join(os.path.dirname(__file__), 'config/streaming_llama2.toml')
        self.model = omniback.init_from_file(config)
        self.request_status = {}
        
        self.continuous_batching  = omniback.get("node.continuous_batching")
        
        self.index = omniback._C.AtomicInt()
        
        # memory
        
        left_mem = torch.cuda.mem_get_info()[0]/(1024.0**2)
        self.total_mem = torch.cuda.mem_get_info()[1]/(1024.0**2)
        
        self.factor = 0.05
        self.need_more_pages = left_mem >= self.total_mem *self.factor
        self.mem_per_page = 2 * self.page_size * 32 * 128 * 2/(1024.0**2) * get_num_layers()
        
        print(f"+---------------------+")
        print(f"| {'Free Memory':<14} | {left_mem:>10.2f} MB |")
        print(f"| {'Total Memory':<14} | {self.total_mem:>10.2f} MB |")
        print(f"| {'Memory per Page':<14} | {self.mem_per_page:>10.2f} MB |")
        print(f"| {'Factor':<14} | {self.factor:>10.2f} |")
        print(f"+---------------------+")
        
        self.need_more_pages = False # todo
        
        self.clean_up = clean_up
        
        
       
        

    def add_request(self, *args, **kwargs):
        # Implement the logic to handle the request asynchronously
        
        request_callback = kwargs["callback"]
        request_id = kwargs.pop("request_id", None)
        if request_id is None:
            id = self.index.increment()
            request_id = f'{id}'#f"cmpl-{shortuuid.random()}"
        else:
            print(f"reusing request_id: {request_id}")
        
        if  self.need_more_pages and self.page_table.available_pages() < 4096/self.page_size:
            free_mem = torch.cuda.mem_get_info()[0]/(1024.0**2)
            mem_can_use = (free_mem - self.total_mem *self.factor)
            self.need_more_pages = mem_can_use >= 0
            if self.need_more_pages:
                
                new_pages = mem_can_use / (self.mem_per_page)
                print(f"mem_can_use: {mem_can_use} MB, new_pages={new_pages}")
                self.page_table.add_more_page(int(new_pages))
                
                current_memory = torch.cuda.memory_allocated()  
                print(f"当前显存占用: {current_memory / 1024**2:.2f} MB")
                cached_memory = torch.cuda.memory_reserved()  
                print(f"缓存显存: {cached_memory / 1024**2:.2f} MB")
        
        input_ids = kwargs.pop("input_ids", None)
        sampling_params: SamplingParams = kwargs.pop("sampling_params") 
        if input_ids is None:
            prompt = kwargs.pop("prompt")
            
            inputs = self.old_tokenizer(prompt, return_tensors="pt", return_attention_mask=False)
            input_ids = inputs['input_ids'][0]
            # input_ids = torch.Tensor(self.tokenizer.encode(prompt).ids)
        
        
        timestamp = 0.0
        old_request_id = kwargs.pop("old_request_id", None)
        if old_request_id:
            print('restarted: CustomBackendEngine: ', args, kwargs, request_id)
            timestamp = kwargs.pop("timestamp", omniback.timestamp())
        else:
            timestamp = omniback.timestamp()
            
        # print(f'input_ids ', type(input_ids))
        event = omniback.Event()
        io = omniback.Dict({
            'data': input_ids,
            'node_name': 'embed_token',
            'event': event,
        })
        io[omniback.TASK_REQUEST_ID_KEY] = request_id
        io[omniback.TASK_MSG_KEY] = omniback.TypedDict({"req_tokens": len(input_ids),
                                                "max_tokens": sampling_params.max_tokens,
                                                "context_length":4096,
                                                "timestamp":timestamp})
        # print("io ", io)
        
        self.request_status[request_id] = RequestState(event, request_callback, sampling_params, args, kwargs, input_ids, [])
        self.request_status[request_id].time = timestamp
            
        event.set_final_callback(lambda : self.final_callback(request_id))
        self.model(io)
    

        
    def final_callback(self, request_id):
        status = self.request_status.pop(request_id)
        
        output = RequestOutput()
        
        try:
            status.event.try_throw()
        except Exception as e:
            omniback.print('into exception handle')
            self.clean_up(request_id)
            
            output.status = Status(StatusCode.UNKNOWN, str(e))
            output.usage = None
            output.finished = True
            omniback.print(f"ERROR MSG: \n{e}")
            # no need
            # self.finish_continuous_batching(request_id)
            status.request_callback(output)
        else:
            omniback.print('no exception')
            self.clean_up(request_id)

            self.continuous_batching_action(request_id)
            if status.dropped:
                # print(type(status.req_tokens), type(status.new_tokens), flush=True)
                status.req_tokens = torch.cat([status.req_tokens]+status.new_tokens)
                # print( (status.req_tokens.shape),  (status.new_tokens.shape), flush=True)
                status.kwargs["input_ids"] = status.req_tokens
                if status.sampling_params.max_tokens != 0:
                    assert status.sampling_params.max_tokens > len(status.new_tokens),  f'{status.sampling_params.max_tokens} >{ len(status.new_tokens)}'
                    status.sampling_params.max_tokens -= len(status.new_tokens)
                status.kwargs['sampling_params'] = status.sampling_params
                # status.kwargs['request_id'] = request_id
                print(f"status.dropped restarted: {request_id}")
                status.kwargs['old_request_id'] = request_id
                status.kwargs['timestamp'] = status.time
                self.add_request(*status.args, **status.kwargs)
        
    def forward(self, ios: List[omniback.Dict]):
        
        for io in ios:
            request_id = io['request_id']#.decode('utf-8')
             # _, request_callback, sampling_params 
            status = self.request_status[request_id]
            
            data = io['data'].cpu()
            status.new_tokens.append(data)
            data_item = data.item()
            
            decode_stream = status.decode_stream
            # text = self.tokenizer.decode(io['data'], skip_special_tokens=True)
            text = decode_stream.step(self.tokenizer, data_item)
            
            
            seq = SequenceOutput()
            output = RequestOutput()
            output.status = Status(StatusCode.OK, "OK")
            output.usage = None
            
            finish_reason = None

            if data_item in status.sampling_params.stop_token_ids or data_item == self.eos_token_id: # eos
                finish_reason = "stop"
                seq.finish_reason = "stop"
                output.finished = True
                seq.text = ""
                print(f"{request_id} - STOP TOKEN ID {data_item} \n")
            elif "finish_reason" in io:
                finish_reason = io["finish_reason"]
                if finish_reason == "length":
                    seq.finish_reason = io["finish_reason"]
                    output.finished = True
                elif finish_reason == "no_page":
                    output.finished = False 
                    status.dropped = True
                else:
                    raise RuntimeError("unsupported reason: "+finish_reason)
                omniback.print(f"{request_id} - finish/restart because of `{finish_reason}`")
                assert not io.contains("restart")  
                if text is None:
                    seq.text = ""
                else:
                    seq.text = text
            
            elif text is None:
                omniback.print(f"id={request_id} - tok={data_item} - no text")
                # todo: 29871? SPIECE_UNDERLINE / 
                output.finished = False 
                seq.text = ''
            else:
                output.finished = False 
                seq.text = text
                
            output.outputs = [seq]
            
            if finish_reason:
                io['finish_reason'] = finish_reason
            status.request_callback(output)
            
            io['result'] = io['data']
        
    
    def continuous_batching_action(self, req_id): # finish pause2prefill pause decode2prefill
        io = omniback.Dict({})
        io[omniback.TASK_REQUEST_ID_KEY] = req_id
        io[omniback.TASK_MSG_KEY] = omniback.TypedDict({"action": "finish"})
        ev = io.set_event()
        ev.set_exception_callback(lambda e: self.python_callback(e)) # would not happen
        self.continuous_batching(io)

    # def finish_continuous_batching(self, req_id):
    #     io = omniback.Dict({})
    #     io[omniback.TASK_REQUEST_ID_KEY] = req_id
    #     io[omniback.TASK_MSG_KEY] = omniback.TypedDict({"action": "finish"})
    #     ev = io.set_event()
    #     ev.set_exception_callback(lambda e: self.python_callback(e)) # would not happen
    #     self.continuous_batching(io)

    def python_callback(self, e):
        # if isinstance(e, RuntimeError):
        #     print(f"Fatal RuntimeError: {e}")
        # else:
        #     print(f"Fatal unknown exception: {e}")
        raise e
    def free_pages(self):
        pass
            

def main(num_layers = 2, max_num_page = 0, port=8000, use_trt=False):
    print( f"CXX11_ABI = {torch._C._GLIBCXX_USE_CXX11_ABI}")
    
    # omniback.init("DebugLogger")
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    if use_trt:
        print('use tensorrt for attn')
        from plain_llama2_trt import set_num_layers, set_page_table, page_size
    else:
        from plain_llama2 import set_num_layers, set_page_table, page_size
    set_num_layers(num_layers)
    
    if max_num_page == 0:
        per_page_mem =  2 * page_size * 32 * 128 * 2/(1024.0**2) * num_layers
        free_mem = torch.cuda.mem_get_info()[0]/(1024.0**2)
        max_num_page = int(free_mem* 0.2 / per_page_mem)
        omniback.print(f"max_num_page: {max_num_page}")
    page_table = set_page_table(max_num_page)
    
    register_engine("llama2", CustomBackendEngine(use_trt=use_trt))

    # fire.Fire(main)
    import sys
    sys.argv = [
        "openai_server_api.py", 
        "--model_id", "my_model",
        "--model", "llama2",
        "--host", "0.0.0.0",
        "--port", f'{port}',
        "--log_level", "info",
    ]
    from torchpipe.serve.openai.openai_server_api import main as server_main
    server_main()
    
if __name__ == '__main__':
    
    import fire
    fire.Fire(main)
