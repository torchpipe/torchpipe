import pytest

from typing import List
import omniback
import time

        
class PY:
    def __init__(self, *args, **kwargs) -> None:
        self.data = {"1,2": 3}
        self.g = {"dsfafwed": self.data}

    def forward(self, inout: List[omniback.Dict]):
        """Process data by copying 'data' to 'result'"""
        inout[0]['result'] = inout[0]['data']


 
if __name__ == "__main__":
    omniback.register('py', PY)
