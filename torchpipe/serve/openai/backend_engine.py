
import queue
import threading
import shortuuid

from scalellm._C import (LLMHandler, Message, Priority, RequestOutput,
                         SamplingParams, Status, StatusCode, SequenceOutput, Usage)

import torchpipe as tp
class BackendEngine:
    def __init__(self, args: dict) -> None:
        self.config = args
        
        self.model = tp.Pipe({'Interpreter::backend':"EventLoop",
                           'backend':"Identity"})
        
    def pre_forward(self, data: dict) -> None:
        return 
    def post_forward(self, data: dict) -> None:
        return
    
    def forward_async(self, data: dict) -> None:
        
        request_callback = data['callback']
        request_id = f"cmpl-{shortuuid.random()}"
        
        
        output = RequestOutput()
        output.prompt = "prompt"
        output.status = Status(StatusCode.OK, "OK")
        
        
        ev = tp.Event()
        input = {'data': data['prompt'], 'event': ev, 'request_id': request_id}
        
        def callback():
            ev.try_throw()
            
            seq = SequenceOutput()
            seq.text = input['result'].decode('utf-8')+ " good"
            seq.index = 0
            seq.logprobs = []
            output.outputs = [seq]
            output.usage = None
            output.finished = True
            
            request_callback(output)
            
        
        ev.set_callback(callback)
        
        
        
    
        
        self.model(input)
        
        # print("d2")
        # while True:
        #     import time
        #     time.sleep(1)
        #     print("d1")
        # ev.Wait()
        print("d3")
        
        
        
        
        
        
        # output = RequestOutput()
        # output.prompt = "prompt"
        # output.status = Status(StatusCode.OK, "OK")
        # seq = SequenceOutput()
        # seq.index = 0
        # seq.text = "Test text"
        # seq.logprobs = []
        # output.outputs = [seq]
        # output.usage = None
        # output.finished = True

    def callback(self) -> bool:
        print('callback')
        pass
