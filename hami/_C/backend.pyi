from typing import Union, Dict, List, Optional
import hami



class Backend:
    def __init__(self, config: Dict[str, str], kwargs: Optional[Dict[str, hami.Any]]) -> None: ...
    def __call__(self, inout: Union[hami.Dict, List[hami.Dict], dict]) -> None: ...
    def max(self) -> int : ...
    def min(self) -> int : ...
    




# Template for register initialized instance
# Usage example:
# a = PyInstance(...)
# hami.register(name="name_1", a)
# Forward[name_1]
# - max and min are optional; max defaults to maximum, min defaults to 1
# - `init`` is optional and can be done inside python code. once defined, you can use it to initialize the instance:
# Init[name_1] or Launch[name_1]
# - multiple registered instances: Forward[name_1, name_2, name_3]
# - hami will make a copy of the instance to keep it alive. Release it by hami.unregister('name_1') if necessary.
class PyInstance:
    def forward(self, inout: List[hami.Dict]) -> None:
        """
        Processes a list of dictionaries.

        :param inout: List of dictionaries to be processed.
        """
        ...

    def max(self) -> int : ...
    def min(self) -> int : ...
    
    def init(self, config: Dict[str, str], kwargs: Optional[hami.Dict] = None) -> None: ...
