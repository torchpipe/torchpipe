


import os, sys


class BackendEngine:
    def __init__(self):
        """
        Initialize the BackendEngine.
        """
        pass
    def forward_async(self, *args, **kwargs):
        """
        Forward the request to the backend engine asynchronously.
        """
        raise NotImplementedError("This method should be implemented in the derived class.")
    
_registered_engines = {}

def register_backend_engine(model_id: str, backend_engine: BackendEngine) -> None:
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