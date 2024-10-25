from enum import Enum
from typing import List, Optional

class Usage:
    def __init__(self) -> None:
        self.num_prompt_tokens: int = 0
        self.num_generated_tokens: int = 0
        self.num_total_tokens: int = 0

    def __repr__(self) -> str:
        return (f"Usage(num_prompt_tokens={self.num_prompt_tokens}, "
                f"num_generated_tokens={self.num_generated_tokens}, "
                f"num_total_tokens={self.num_total_tokens})")

class LogProbData:
    def __init__(self) -> None:
        self.token: str = ""
        self.token_id: int = 0
        self.logprob: float = 0.0
        self.finished_token: bool = False

    def __repr__(self) -> str:
        return (f"LogProbData(token={self.token}, token_id={self.token_id}, "
                f"logprob={self.logprob}, finished_token={self.finished_token})")

class LogProb:
    def __init__(self) -> None:
        self.token: str = ""
        self.token_id: int = 0
        self.logprob: float = 0.0
        self.finished_token: bool = False
        self.top_logprobs: Optional[List[LogProbData]] = None

    def __repr__(self) -> str:
        return (f"LogProb(token={self.token}, token_id={self.token_id}, "
                f"logprob={self.logprob}, finished_token={self.finished_token}, "
                f"top_logprobs={self.top_logprobs})")

class SequenceOutput:
    def __init__(self) -> None:
        self.index: int = 0
        self.text: str = ""
        self.token_ids: List[int] = []
        self.finish_reason: Optional[str] = None
        self.logprobs: Optional[List[LogProb]] = None

    def __repr__(self) -> str:
        return (f"SequenceOutput(index={self.index}, text={self.text}, "
                f"token_ids={self.token_ids}, finish_reason={self.finish_reason}, "
                f"logprobs={self.logprobs})")

class RequestOutput:
    def __init__(self) -> None:
        self.prompt: Optional[str] = None
        self.status: Optional['Status'] = None
        self.outputs: List[SequenceOutput] = []
        self.usage: Optional[Usage] = None
        self.finished: bool = False

    def __repr__(self) -> str:
        return (f"RequestOutput(prompt={self.prompt}, status={self.status}, "
                f"outputs={self.outputs}, usage={self.usage}, finished={self.finished})")

class StatusCode(Enum):
    OK = 0
    CANCELLED = 1
    UNKNOWN = 2
    INVALID_ARGUMENT = 3
    DEADLINE_EXCEEDED = 4
    RESOURCE_EXHAUSTED = 5
    UNAUTHENTICATED = 6
    UNAVAILABLE = 7
    UNIMPLEMENTED = 8

class Status:
    def __init__(self, code: StatusCode, message: str) -> None:
        self._code = code
        self._message = message

    def __repr__(self) -> str:
        return f"Status(code={self._code}, message={self._message})"

    @property
    def code(self) -> StatusCode:
        return self._code

    @property
    def message(self) -> str:
        return self._message

    @property
    def ok(self) -> bool:
        return self._code == StatusCode.OK