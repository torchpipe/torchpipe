



import os
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Sequence, Union, Tuple


_model_entrypoints : Dict[str, Callable[..., Any]] = {}  # mapping of model names to architecture entrypoint fns


"""
    Examples:
    >>> # define a registry
    >>> MODELS = Registry('models')
    >>> # registry the `ResNet` to `MODELS`
    >>> # examples:
    >>> @MODELS.register
        def resnet50(pretrained=False, **kwargs):
    >>>     class ResNet:
    >>>         pass
    >>>     return ResNet()
    >>> # or 带参数的注册:
    >>> @MODELS.register("resnet50")
        def resnet50(pretrained=False, **kwargs):
    >>>     class ResNet:
    >>>         pass
    >>>     return ResNet()
"""



def _register_model(model_fn=None, model_name=None, ):

    if not callable(model_fn):
        raise TypeError(f'model must be Callable, but got {type(model_fn)}')
    
    if model_name is None:
        model_name = model_fn.__name__

    if not isinstance(model_name, str):
        raise TypeError(f'model_name must be str, but got {type(model_name)}')


    if  model_name in _model_entrypoints:
        existed_model = _model_entrypoints[model_name]
        raise KeyError(f'{model_name} is already registered '
                    f'at {existed_model.__module__}')
    
    _model_entrypoints[model_name] = model_fn

    print(f"register {model_name} successfully")



def _register(model_fn=None, model_name=None):

    if model_fn is not None:
        _register_model(model_name=model_name, model_fn=model_fn)
        return model_fn

    # use it as a decorator: @register_model("resnet50")
    def _register(model_fn):
        _register_model(model_name=model_name, model_fn=model_fn)
        return model_fn

    return _register


## 支持两种注册方式，@register("resnet50") 和 @register
def register_model(param):
    if isinstance(param, str):
        return _register(model_name=param)
    elif callable(param):
        return _register(model_fn=param)
    else:
        raise TypeError(
            'model must be either a str or callable, '
            f'but got {type(param)}')

## 对于timm的模型，可以直接注册
def register_model_from_timm(model_name=None):
    if model_name is None:
        raise ValueError(f"model_name is None")
    
    import timm
    if model_name not in timm.list_models():
        raise ValueError(f"{model_name} is not supported by timm")
    else:
        fun = lambda **kwargs: timm.create_model(model_name, **kwargs)
        _register_model(model_name=model_name, model_fn=fun)

    

def create_model(model_name, **kwargs):
    if model_name not in _model_entrypoints:
        raise KeyError(f'{model_name} is not registered in registry')
    return _model_entrypoints[model_name](**kwargs)






def list_models():
    return list(_model_entrypoints.keys())




