
import queue
import threading
import shortuuid
import time
import torch
# from scalellm._C import (LLMHandler, Message, Priority, RequestOutput,
#                          SamplingParams, Status, StatusCode, SequenceOutput, Usage)

from torchpipe.serve.output import RequestOutput, SequenceOutput,  Status, StatusCode
import torchpipe as tp

storage = tp.ThreadSafeKVStorage.getInstance()
 
request_state = {}



class PyStream:
    def init(self, config) -> bool:
        from transformers import AutoTokenizer, AutoModelForCausalLM

        # self.model = tp.Pipe("config/llama2_streaming.toml")
        # print("BackendEngine initialized")
        self.tokenizer = AutoTokenizer.from_pretrained('model_files/')
        print(f"Initalized PyStream with config:",  config)
        return True
    
    def forward(self, input: tp._C.Dict) -> None:
        print(list(input.keys()))
        print("PyStream ", input['data'].shape)
        request_id = input['request_id'].decode('utf-8')
        print('streaming with request_id = ', request_id)
        # print(input['input_tokens_result'])
        
        seq = SequenceOutput()
        
        output = RequestOutput()
        output.prompt = "prompt"
        output.status = Status(StatusCode.OK, "OK")
        
        output.usage = None
        
        if input['key'] == 2:
            seq.finish_reason = "stop"
            output.finished = True 
        else:
            
            # start = time.perf_counter()
            text = self.tokenizer.decode(input['key'], skip_special_tokens=True)
            # print(f"Time taken to decode: {time.perf_counter() - start}")
            print(f"text = ", text, input['key'])
            
            seq.text = text  # input['result'].decode('utf-8')+ " good"
            seq.index = 0
            if not "input_tokens" in input:
                seq.finish_reason = "length"
                output.finished = True
            else:
                output.finished = False 
            
            # seq.text += str(output.finished)
        output.outputs = [seq]
        
        _, local_cb = request_state[request_id]
        
        
        local_cb(output)
    
tp.register_backend( PyStream, "PyStream")



class BackendEngine:
    def __init__(self) -> None:
        from transformers import AutoTokenizer, AutoModelForCausalLM

        self.model = tp.Pipe("config/llama2_streaming.toml")
        print("BackendEngine initialized")
        self.tokenizer = AutoTokenizer.from_pretrained('model_files/')
        
            
    def forward_async(self, data: dict) -> None:
        
        request_callback = data['callback']
        request_id = f"cmpl-{shortuuid.random()}"
        
        ev = tp.Event()
        input = data['prompt']#{'data': data['prompt'], 'event': ev, 'request_id': request_id}
        max_tokens = data['max_tokens']
        
        # start = time.perf_counter()
        inputs = self.tokenizer(input, return_tensors="pt")
        # print(f"Time taken to tokenize: {time.perf_counter() - start}")
        # print(inputs["input_ids"])
        print(f'max_tokens={max_tokens}')
        inputs = {
            'data': inputs["input_ids"][0],
            'node_name': 'input',
            'trt_plugin': 'batchless_prefill',
            'event': ev, 'request_id': request_id,
            "max_tokens":max_tokens, 
        }
        
        request_state[request_id] = (ev, request_callback)
            
        def finish_cb():
            print('finish_cb')
            local_ev, local_cb = request_state[request_id]
            
            output = RequestOutput()
            
            
            
            try:
                local_ev.try_throw()
            except Exception as e:
                print('exception')
                
                output.status = Status(StatusCode.UNKNOWN, str(e))
                output.usage = None
                output.finished = True
                print(e)
                storage.remove(request_id)
                local_cb(output)
            else:
                print('no exception')
                
                # output.status = Status(StatusCode.OK, "OK")
                # output.outputs = []
                # output.usage = None
                # output.finished = True
                
            del request_state[request_id]
            
            
        ev.set_callback(finish_cb)

        
        total_memory = torch.cuda.get_device_properties(0).total_memory
        left = (total_memory - torch.cuda.memory_reserved(0)) / 1024 / 1024
        while (left < 1024):
            # torch.cuda.empty_cache()
            print("Waiting for memory. left = ", left)
            time.sleep(0.01)
            
            
        self.model(inputs)