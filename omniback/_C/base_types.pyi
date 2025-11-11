
from enum import Enum
from typing import Callable, List, Optional, Any, Dict

from omniback._C import Any, dict, Backend, create, Event



# Defined in csrc/*.cpp
# class any:
#     def __init__(self, value: Any) -> None: ...

# class dict:
#     def __init__(self, value: Any) -> None: ...
# class backend:
#     def __init__(self, name: str, value: Any) -> None: ...    
def create(name: str, register_name: Optional[str]) -> Backend: ...

def init(backend: str, register_name: Optional[str]) -> Backend: ...

