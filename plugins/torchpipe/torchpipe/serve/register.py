


import os, sys
from abc import ABC, abstractmethod
import omniback
from typing import List

class BackendEngine(ABC):
    @abstractmethod
    def add_request(self, *args, **kwargs):
        pass
    
    @abstractmethod
    def forward(self, ios:List[omniback.Dict]):
        pass

    
_registered_engines = {}

def register_engine(model_id: str, backend_engine: BackendEngine) -> None:
    """
    Register the backend engine for torchpipe.
    """
    if model_id in _registered_engines:
        raise ValueError(f"Backend engine for model_id '{model_id}' is already registered.")
    _registered_engines[model_id] = backend_engine
    print(f"Backend engine for model_id '{model_id}' registered successfully.")

def get_backend_engine(model_id: str) -> BackendEngine:
    """
    Get the registered backend engine for the given model_id.
    """
    if model_id not in _registered_engines:
        raise ValueError(f"Backend engine for model_id '{model_id}' is not registered.")
    return _registered_engines[model_id]