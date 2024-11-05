import asyncio
import os
import queue
from typing import List, Optional

# from scalellm._C import (LLMHandler, Message, Priority,
#                         Status, StatusCode, SequenceOutput, Usage)
# from scalellm.downloader import download_hf_model

from torchpipe.serve.output import RequestOutput

from torchpipe.serve.errors import ValidationError
from torchpipe.serve.api_protocol import CompletionRequest
import sys
import threading
 


class OutputAsyncStream:
    """A stream of RequestOutput objects, which can be used to
    send responses to the client asynchronously."""

    def __init__(self) -> None:
        # asyncio.Queue is used to store the items in the stream
        self._queue = asyncio.Queue()
        self._cancelled = False
        self.loop = asyncio.get_event_loop()
        
 

    # 异步方法，将项目放入队列中
    async def async_put(self, item: RequestOutput) -> bool:
        # if the stream is cancelled, return False
        if self._cancelled:
            return False

        if item.status is not None and not item.status.ok:
            await self._queue.put(
                ValidationError(item.status.code, item.status.message)
            )
            return False

        # put the item into the queue
        await self._queue.put(item)
        if item.finished:
            await self._queue.put(StopAsyncIteration())
        return True

    # 同步方法，使用线程安全的方式执行 async_put，用于回调的接口
    def put(self, item: RequestOutput) -> bool:
        
        future = asyncio.run_coroutine_threadsafe(self.async_put(item), self.loop)
        return future.result()
    
    
    # report an error to the stream, rerais as an exception
    def error(self, error: str) -> bool:
        self._queue.put_nowait(Exception(error))
        return True

    # cancel the stream
    def cancel(self) -> None:
        self._cancelled = True
        self._queue.put_nowait(StopAsyncIteration())

    def __aiter__(self):
        return self

    # async generator to iterate over the stream
    async def __anext__(self) -> RequestOutput:
        item = await self._queue.get()
        # reraise the exception
        if isinstance(item, Exception):
            raise item
        return item

# from torchpipe.serve.api_protocol import (ChatCompletionRequest,
#                                          CompletionRequest)

# 获取环境变量 BACKEND_ENGINE_PATH 的值
backend_engine_path = os.getenv('BACKEND_ENGINE_PATH', './')

# 将 BACKEND_ENGINE_PATH 添加到 sys.path
sys.path.append(backend_engine_path)

# 导入 BackendEngine 模块
try:
    from backend_engine import BackendEngine
except ImportError:
    raise ImportError(
        "The BackendEngine module is not found. Please make sure that the BACKEND_ENGINE_PATH environment variable is set correctly."
    )
        
class AsyncEngine:
    def __init__(
        self,
        args: dict,
    ) -> None:
        self.config = args
        self.backend = BackendEngine()

    # schedule a request to the engine, and return a stream to receive output
    async def schedule_async(
        self,
        prompt: str,
        sampling_params,
        priority: None,
        stream: bool = False,
        request: CompletionRequest = None,
    ) -> OutputAsyncStream:
        output_stream = OutputAsyncStream()

        def callback(output: RequestOutput) -> bool:
            output.prompt = prompt
            return output_stream.put(output)

        # use default sampling parameters if not provided
        # self._handler.schedule_async(
        #     prompt, sampling_params, priority, stream, callback
        # )
        try:
            await self.backend.forward_async({'prompt': prompt, 'priority': priority,
                                        'stream': stream, 'callback': callback,
                                        'max_tokens': request.max_tokens})
        except Exception as e:
            output_stream.error(str(e))
        return output_stream
    
    # start the engine, non-blocking
    def start(self) -> None:
        pass
        # return self._handler.start()

    # stop the engine, non-blocking
    def stop(self) -> None:
        pass
        # return self._handler.stop()
 
 
    def __del__(self):
        if hasattr(self, "_handler"):
            self._handler.reset()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()
        self.__del__()
        return False
    
    def __repr__(self) -> str:
        if self._draft_model:
            return f"AsyncEngine(model={self._model}, draft_model={self._draft_model})"
        return f"AsyncEngine(model={self._model})"
