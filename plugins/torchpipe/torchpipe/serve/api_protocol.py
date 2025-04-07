# Adapted from https://github.com/lm-sys/FastChat and sglang
import time
from typing import Dict, List, Literal, Optional, Union

import shortuuid
from pydantic import BaseModel, Field


class ErrorResponse(BaseModel):
    message: Optional[str] = None
    code: Optional[int] = None


class ModelPermission(BaseModel):
    id: str = Field(default_factory=lambda: f"modelperm-{shortuuid.random()}")
    object: str = "model_permission"
    created: int = Field(default_factory=lambda: int(time.time()))
    allow_create_engine: bool = False
    allow_sampling: bool = True
    allow_logprobs: bool = True
    allow_search_indices: bool = False
    allow_view: bool = True
    allow_fine_tuning: bool = False
    organization: str = "*"
    group: Optional[str] = None
    is_blocking: bool = False


class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = ""
    root: Optional[str] = None
    parent: Optional[str] = None
    permission: List[ModelPermission] = []


class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelCard] = []


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    total_tokens: int = 0
    completion_tokens: Optional[int] = 0


class ChatCompletionLogProbData(BaseModel):
    token: str
    token_id: int
    logprob: float
    bytes: Optional[List[int]] = None


class ChatCompletionLogProb(ChatCompletionLogProbData):
    top_logprobs: Optional[List[ChatCompletionLogProbData]] = None


class ChatCompletionLogProbs(BaseModel):
    content: Optional[List[ChatCompletionLogProb]] = None


# class ChatCompletionMessage(BaseModel):
#     role: Literal["user", "system", "assistant"]
#     content: str


class ChatCompletionMessageContentTextPart(BaseModel):
    type: Literal["text"]
    text: str


class ChatCompletionMessageContentImageURL(BaseModel):
    url: str
    detail: Optional[Literal["auto", "low", "high"]] = "auto"


class ChatCompletionMessageContentImagePart(BaseModel):
    type: Literal["image_url"]
    image_url: ChatCompletionMessageContentImageURL
    modalities: Optional[Literal["image", "multi-images", "video"]] = "image"


ChatCompletionMessageContentPart = Union[
    ChatCompletionMessageContentTextPart, ChatCompletionMessageContentImagePart
]


class ChatCompletionMessageGenericParam(BaseModel):
    role: Literal["system", "assistant", "tool"]
    content: Union[str, List[ChatCompletionMessageContentTextPart]]


class ChatCompletionMessageUserParam(BaseModel):
    role: Literal["user"]
    content: Union[str, List[ChatCompletionMessageContentPart]]


class StreamOptions(BaseModel):
    include_usage: Optional[bool]

ChatCompletionMessageParam = Union[
    ChatCompletionMessageGenericParam, ChatCompletionMessageUserParam
]

class JsonSchemaResponseFormat(BaseModel):
    name: str
    description: Optional[str] = None
    # use alias to workaround pydantic conflict
    schema_: Optional[Dict[str, object]] = Field(alias="schema", default=None)
    strict: Optional[bool] = False

class ResponseFormat(BaseModel):
    type: Literal["text", "json_object", "json_schema"]
    json_schema: Optional[JsonSchemaResponseFormat] = None

class ChatCompletionRequest(BaseModel):
    # Ordered by official OpenAI API documentation
    # https://platform.openai.com/docs/api-reference/chat/create
    # messages: List[ChatCompletionMessageParam]
    messages: List[Dict[str, Union[str, List[Dict[str, Union[str, Dict[str, str]]]]]]]
    
#       "messages": [
#     {
#       "role": "user",
#       "content": [
#         {
#           "type": "text",
#           "text": "What’s in this image?"
#         },
#         {
#           "type": "image_url",
#           "image_url": {
#             "url": f"data:image/jpeg;base64,{base64_image}"
#           }
#         }
#       ]
#     }
#   ],
    model: str
    frequency_penalty: float = 0.0
    logit_bias: Optional[Dict[str, float]] = None
    logprobs: bool = False
    top_logprobs: Optional[int] = None
    max_tokens: Optional[int] = None
    n: int = 1
    presence_penalty: float = 0.0
    response_format: Optional[ResponseFormat] = None
    seed: Optional[int] = None
    stop: Optional[Union[str, List[str]]] = None
    stream: bool = False
    stream_options: Optional[StreamOptions] = None
    temperature: float = 0.7
    top_p: float = 1.0
    user: Optional[str] = None




class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessage
    logprobs: Optional[ChatCompletionLogProbs] = None
    finish_reason: Optional[Literal["stop", "length"]] = None


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{shortuuid.random()}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionResponseChoice]
    usage: Optional[UsageInfo] = None


class DeltaMessage(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None


class ChatCompletionResponseStreamChoice(BaseModel):
    index: int
    delta: DeltaMessage
    logprobs: Optional[ChatCompletionLogProbs] = None
    finish_reason: Optional[Literal["stop", "length"]] = None


class ChatCompletionStreamResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{shortuuid.random()}")
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionResponseStreamChoice]
    usage: Optional[UsageInfo] = None


class CompletionRequest(BaseModel):
    model: str
    prompt: str
    priority: Optional[Literal["default", "low", "normal", "high"]] = None
    # suffix: Optional[str] = None
    n: Optional[int] = 1
    best_of: Optional[int] = None
    max_tokens: Optional[int] = 16
    stream: Optional[bool] = False
    stream_options: Optional[StreamOptions] = None
    logprobs: Optional[int] = None
    echo: Optional[bool] = False
    temperature: Optional[float] = 0.7
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    repetition_penalty: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    top_k: Optional[int] = -1
    # user: Optional[str] = None
    skip_special_tokens: Optional[bool] = True
    ignore_eos: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    stop_token_ids: Optional[List[int]] = None
    # seed: Optional[int] = None


class CompletionLogProbs(BaseModel):
    text_offset: List[int] = Field(default_factory=list)
    token_logprobs: List[Optional[float]] = Field(default_factory=list)
    tokens: List[str] = Field(default_factory=list)
    token_ids: List[int] = Field(default_factory=list)
    top_logprobs: Optional[List[Optional[Dict[str, float]]]] = None


class CompletionResponseChoice(BaseModel):
    index: int
    text: str
    logprobs: Optional[CompletionLogProbs] = None
    finish_reason: Optional[Literal["stop", "length"]] = None


class CompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"cmpl-{shortuuid.random()}")
    object: str = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[CompletionResponseChoice]
    usage: Optional[UsageInfo] = None


class CompletionResponseStreamChoice(BaseModel):
    index: int
    text: str
    logprobs: Optional[CompletionLogProbs] = None
    finish_reason: Optional[Literal["stop", "length"]] = None


class CompletionStreamResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"cmpl-{shortuuid.random()}")
    object: str = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[CompletionResponseStreamChoice]
    usage: Optional[UsageInfo] = None
