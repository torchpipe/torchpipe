# Copyright 2021-2024 NetEase.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Union, Tuple, List
from typing import Tuple, List
from typing import List, Optional, Dict, Any, Dict

# from builtins import isinstance
import torch
import logging
import os

# from .tool import cpp_tools

from torchpipe.libipipe import Interpreter, parse_toml, TASK_RESULT_KEY


class pipe:
    """python interface for the c++ library. A simple wrapper for :ref:`Interpreter <Interpreter>` . Usage:

    .. code-block:: python

        models = pipe({"model":"model_bytes...."})
        input = {TASK_DATA_KEY:torch.from_numpy(...)}
        result : torch.Tensor = input[TASK_RESULT_KEY]

    """

    def __init__(self, config: Union[Dict[str, str], Dict[str, Dict[str, str]], str]):
        """init with configuration.

        :param config: toml file and plain dict are supported. These parameters will be passed to all the backends involved.
        :type config: Dict[str, str] | Dict[str,Dict[str, str]] | str
        """
        self.Interpreter = Interpreter()
        if not config:
            raise RuntimeError(f"empty config : {config}")
        if isinstance(config, dict):
            for k, v in config.items():
                if isinstance(v, dict):
                    for k2, v2 in v.items():
                        if not isinstance(v, (bytes, str)):
                            config[k][k2] = str(v2)  # .encode("utf8")
                else:
                    if not isinstance(v, (bytes, str)):
                        config[k] = str(v)  # .encode("utf8")
            self._init(config)
        else:
            self._init_from_toml(config)

    def _init_from_toml(self, toml_path):
        self.config = parse_toml(toml_path)
        return self.Interpreter.init(self.config)

    def _init(self, config):
        self.config = config
        return self.Interpreter.init(config)

    def __call__(
        self,
        data: Optional["Dict[str, Any] | List[Dict[str, Any]]"] = None,
        **kwargs: Any,
    ) -> Union[None, Any]:
        """thread-safe inference. The input could be a single dict, a list of dict for multiple inputs, or raw key-value pairs.

        :param data: input dict(s), defaults to None
        :type data: `Dict[str, Any] | List[Dict[str, Any]]`, optional
        :return: None if data exists, else Any.
        :rtype: Union[None, Any]
        """
        if isinstance(data, list):
            if len(kwargs):
                for di in data:
                    di.update(kwargs)
            self.Interpreter(data)
        elif isinstance(data, dict):
            data.update(kwargs)
            self.Interpreter(data)
        else:
            data = {}
            data.update(kwargs)
            self.Interpreter(data)
            return data[TASK_RESULT_KEY]

    def max(self):
        return self.Interpreter.max()

    def min(self):
        return self.Interpreter.min()

    def __del__(self):
        self.Interpreter = None
