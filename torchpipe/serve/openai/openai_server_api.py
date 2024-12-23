"""A server that provides OpenAI-compatible RESTful APIs. It supports:

- Chat Completions. (Reference: https://platform.openai.com/docs/api-reference/chat)
- Completions. (Reference: https://platform.openai.com/docs/api-reference/completions)

Usage:
python3 -m scalellm.serve.api_server
"""

import os
from http import HTTPStatus
from pathlib import Path
from typing import Optional, List
import shortuuid
import time

import fastapi
import uvicorn
from fastapi.responses import JSONResponse, Response

from torchpipe.serve.output import LogProb
from torchpipe.serve.errors import ValidationError
# , get_metrics
from torchpipe.serve.api_protocol import (ChatCompletionRequest,
                                         CompletionRequest, ErrorResponse,
                                         ModelCard, ModelList, ModelPermission)
# from scalellm.serve.chat_handler import (generate_chat_response,
#                                          generate_chat_stream_response)
# from scalellm.serve.completion_handler import (
#     generate_completion_response, generate_completion_stream_response)
from torchpipe.serve.streaming_response import SafeStreamingResponse
# from scalellm.serve.common import (get_printable_token, jsonify_model,
#                                    to_api_usage, to_priority)

def jsonify_model(obj):
    return obj.model_dump_json(exclude_unset=True)

def get_printable_token(logprob) -> str:
    return (
        logprob.token
        if logprob.finished_token
        else "".join(
            f"\\x{byte:02x}" for byte in logprob.token.encode("utf-8", errors="replace")
        )
    )
    
from torchpipe.serve.api_protocol import (CompletionLogProbs, CompletionRequest,
                                         CompletionResponse,
                                         CompletionResponseChoice,
                                         CompletionResponseStreamChoice,
                                         CompletionStreamResponse)


from torchpipe.serve.openai.async_backend_engine import AsyncEngine
from torchpipe.serve.server_args import parse_args

app = fastapi.FastAPI(docs_url='/')
llm_engine: AsyncEngine = None
models = None


def create_error_response(
    message: str, code: int, status_code: HTTPStatus = HTTPStatus.BAD_REQUEST
) -> JSONResponse:
    return JSONResponse(
        {"error": ErrorResponse(message=message, code=code).dict()},
        status_code=status_code.value,
    )


def check_model(request) -> Optional[JSONResponse]:
    # if request.model not in models:
    #     return create_error_response(
    #         message=f"The model `{request.model}` does not exist.",
    #         code=404,
    #         status_code=HTTPStatus.NOT_FOUND,
    #     )
    return None


@app.exception_handler(ValidationError)
async def validation_exception_handler(request, e):
    return create_error_response(e.message, e.code)



@app.get("/health")
async def show_health() -> Response:
    return Response(status_code=200)


@app.get("/v1/models")
async def show_available_models():
    model_cards = [ModelCard(id=model_id, permission=[ModelPermission()])]
    return ModelList(data=model_cards)

 

def to_api_logprobs(
    logprobs: Optional[List[LogProb]],
    offset: int,
) -> Optional[CompletionLogProbs]:
    if logprobs is None:
        return None

    text_offset, tokens, token_ids, token_logprobs, top_logprobs = [], [], [], [], []

    for logprob in logprobs:
        text_offset.append(offset)
        offset += len(logprob.token)
        tokens.append(get_printable_token(logprob))
        token_ids.append(logprob.token_id)
        token_logprobs.append(logprob.logprob)

        if logprob.top_logprobs:
            top_logprobs.append(
                {get_printable_token(top): top.logprob for top in logprob.top_logprobs}
            )
        else:
            top_logprobs.append(None)

    return CompletionLogProbs(
        text_offset=text_offset,
        tokens=tokens,
        token_ids=token_ids,
        token_logprobs=token_logprobs,
        top_logprobs=top_logprobs,
    )


from torchpipe.libipipe import SamplingParams
# SamplingParams = tp._C.SamplingParams
def to_sampling_params(request: CompletionRequest) -> SamplingParams:
    sp = SamplingParams()
    sp.max_tokens = request.max_tokens
    sp.n = request.n
    sp.best_of = request.best_of
    sp.echo = request.echo
    sp.frequency_penalty = request.frequency_penalty
    sp.presence_penalty = request.presence_penalty
    sp.repetition_penalty = request.repetition_penalty
    sp.temperature = request.temperature
    sp.top_p = request.top_p
    sp.top_k = request.top_k
    if request.logprobs:
        sp.logprobs = True
        sp.top_logprobs = request.logprobs
    sp.skip_special_tokens = request.skip_special_tokens
    if isinstance(request.stop, str):
        sp.stop = [request.stop]
    else:
        sp.stop = request.stop
    sp.ignore_eos = request.ignore_eos
    sp.stop_token_ids = request.stop_token_ids
    return sp

async def generate_completion_stream_response(
    request: CompletionRequest, engine: AsyncEngine
) -> SafeStreamingResponse:
    assert request.stream, "non-streaming request is not supported"

    request_id = f"cmpl-{shortuuid.random()}"
    created_time = int(time.time())
    chunk_object_type = "text_completion"
    model = request.model

    sampling_params = to_sampling_params(request)
    # priority = to_priority(request.priority)
    output_stream = await engine.schedule_async(
        request.prompt,
        sampling_params=sampling_params,
        priority=None,
        stream=request.stream,
        request= request,
    )

    include_usage = request.stream_options and request.stream_options.include_usage

    async def generate_stream_content():
        prompt_len = len(request.prompt)
        offsets = {}
        usage = None
        async for output in output_stream:
            for seq_output in output.outputs:
                cur_offset = offsets.setdefault(seq_output.index, prompt_len)
                offsets[seq_output.index] += len(seq_output.text)
                # send chunk with delta message
                if seq_output.text:
                    response = CompletionStreamResponse(
                        id=request_id,
                        object=chunk_object_type,
                        created=created_time,
                        model=model,
                        choices=[
                            CompletionResponseStreamChoice(
                                index=seq_output.index,
                                text=seq_output.text,
                                logprobs=to_api_logprobs(
                                    seq_output.logprobs, cur_offset
                                ),
                                finish_reason=seq_output.finish_reason,
                            )
                        ],
                    )
                    if include_usage:
                        response.usage = None
                    yield f"data: {jsonify_model(response)}\n\n"

            # record last usage info
            if output.usage:
                usage = output.usage

        # send additional chunk for usage info
        if include_usage and usage:
            response = CompletionStreamResponse(
                id=request_id,
                object=chunk_object_type,
                created=created_time,
                model=model,
                choices=[],
                usage=None,
            )
            yield f"data: {jsonify_model(response)}\n\n"
        yield "data: [DONE]\n\n"

    return SafeStreamingResponse(
        content=generate_stream_content(), media_type="text/event-stream"
    )

@app.post("/v1/completions")
async def create_completion(request: CompletionRequest):
    """Creates a completion for the prompt"""
    error_response = check_model(request)
    if error_response is not None:
        return error_response

    # results cannot be streamed when best_of != n

    if request.stream:
        return await generate_completion_stream_response(request, llm_engine)
    return create_error_response(
            message=f"non-streaming completions are not supported yet",
            code=404,
            status_code=HTTPStatus.METHOD_NOT_ALLOWED,
        )

    return await generate_completion_response(request, llm_engine)


 
def main():
    global models, llm_engine
    args = parse_args()
    # set the model_id
    if args.model_id is not None:
        #  use the model_id provided by the user
        model_id = args.model_id
    elif os.path.exists(args.model):
        # use the directory name of the model path
        model_id = Path(args.model).stem
    else:
        # model is model name
        model_id = args.model
    models = [model_id, 'string', '']

    # initialize the LLM engine
    llm_engine = AsyncEngine(
        {"model":args.model}
    )

    try:
        llm_engine.start()
        print("Starting uvicorn...")
        uvicorn.run(
            app,
            host=args.host,
            port=args.port,
            log_level=args.log_level,
            ssl_keyfile=args.ssl_keyfile,
            ssl_certfile=args.ssl_certfile,
        )
    except KeyboardInterrupt:
        pass
    finally:
        # stop the LLM engine
        llm_engine.stop()
    print(f"Server stop.")

if __name__ == "__main__":
    main()