import os
import time
import torchpipe
# import torchpipe.utils.test
from typing import List
import pytest
class TestBackend:
    @classmethod
    def setup_class(self):
        self.dict  = torchpipe._C.TypedDict({"1": 2})
        
    def test_int(self):
        assert self.dict['1'] == 2
        self.dict['12.']=-1
        assert self.dict['12.'] == -1
        
        with pytest.raises(RuntimeError):
            self.dict['11.']=-1844674407370955
            print(self.dict)
        # assert self.dict['11.'] == -1844674407370955
        
        # self.dict['1.'] = 2
        # assert self.dict['1.'] == 2
    def test_nest(self):
        b = torchpipe._C.TypedDict({"1": 2})
        self.dict["b"] = b
        with pytest.raises(ValueError):
            self.dict['c'] = self.dict
        print(self.dict.keys())
        
if __name__ == "__main__":
    import time
    time.sleep(2)
    a=TestBackend()
    a.setup_class()
    # a.test_int()
    a.test_nest()
 