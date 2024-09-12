from fastapi import Request, BackgroundTasks
from typing import Dict, Any

from fastchat.conversation import get_conv_template
from fastchat.model.model_adapter import get_conversation_template
from fastapi.responses import StreamingResponse, JSONResponse

import threading

global_config = {
    'conv_template': 'llama-2',
    'model_path': '',
    'context_length': 4096
}

class LocalWorker :
    context_length = global_config["context_length"]

    def __init__(self):
        conv_template, model_path = global_config['conv_template'], global_config['model_path']
        self.conv = self.make_conv_template(conv_template, model_path)
        
        self.current_id = 0
        self.lock = threading.Lock()
    
    def count_token(self, prompt: str, model: str):
        return len(prompt.split())
    
    def make_conv_template(self, conv_template, model_path):
        if conv_template:
            conv = get_conv_template(conv_template)
        else:
            conv = get_conversation_template(model_path)
        return conv
    async def abort(self, request_id):
        print(f"aborting request {request_id}")
        pass
    
    async def generate_stream(self, params):
        context = params.pop("prompt")
        request_id = params.pop("request_id")
        temperature = float(params.get("temperature", 1.0))
        top_p = float(params.get("top_p", 1.0))
        top_k = params.get("top_k", -1.0)
        presence_penalty = float(params.get("presence_penalty", 0.0))
        frequency_penalty = float(params.get("frequency_penalty", 0.0))
        max_new_tokens = params.get("max_new_tokens", 256)
        stop_str = params.get("stop", None)
        # stop_token_ids = params.get("stop_token_ids", None) or []
        # if self.tokenizer.eos_token_id is not None:
        #     stop_token_ids.append(self.tokenizer.eos_token_id)
        echo = params.get("echo", True)
        use_beam_search = params.get("use_beam_search", False)
        best_of = params.get("best_of", None)

        request = params.get("request", None)
        prompt_tokens = 1
        completion_tokens = 0
        
        total = 10
        tst = f"this is text_outputs for test "
        for i in range(total):
            completion_tokens += 1
            tst += f"{i}/{total}; "
            ret = {
                    "text": tst,
                    "error_code": 0,
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens ,
                        "total_tokens": prompt_tokens + completion_tokens,
                    },
                    "cumulative_logprob": [
                        
                    ],
                    "finish_reason": None,
                }
            yield ret
    
    def unique_request_id(self):
        with self.lock:
            id = self.current_id
            self.current_id += 1
        return id
        
worker = LocalWorker()

 
async def count_token(data: Dict[str, Any]):
    prompt = data.pop("prompt")
    return {"count":  worker.count_token(prompt, '')}

async def model_details(data: Dict[str, Any]):
    return {"context_length": worker.context_length}

async def list_models(data: Dict[str, Any]):
    return {"models": ['default', '', 'string']}

async def worker_get_conv_template(data: Dict[str, Any]):
    return {"conv": worker.conv}

async def worker_generate(data: Dict[str, Any]):
    # prompt = data.pop("prompt")
    # return {"response": worker.conv.generate(prompt)}

    # params["request"] = request
    # output = await worker.generate(data)
    # print(data.keys())
    # print(data["request_id"])
    prompt_tokens = 1
    completion_tokens = 1
    ret = {
            "text": "text_outputs",
            "error_code": 0,
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
            "cumulative_logprob": [
                
            ],
            "finish_reason": None,
        }
    
    # worker.abort(data["request_id"])
        # Emit twice here to ensure a 'finish_reason' with empty content in the OpenAI API response.
        # This aligns with the behavior of model_worker.
    return ret


 

def create_background_tasks(request_id):
    async def abort_request() -> None:
        await worker.abort(request_id)

    background_tasks = BackgroundTasks()
    # background_tasks.add_task(release_worker_semaphore)
    background_tasks.add_task(abort_request)
    return background_tasks

async def worker_generate_stream(data: Dict[str, Any]):
    # request_id = g_id_gen.generate_id()
    # data = await params.json()
    request_id =data.pop('request_id', worker.unique_request_id())
    data["request_id"] = request_id
    # data["request"] = params
    generator = worker.generate_stream(data)
    async for item in generator:
        yield item
    # 在迭代器执行完后开始背景任务
    # await worker.abort(request_id)

 
    # return JSONResponse(data)