import omniback
import tvm_ffi
import pytest

def test_queue():
    q = omniback.ffi.Queue()
    print(q.size())
    q.put(q)
    q.put({3: "1"})
    q.put({11: {1: 4}})

    re = [q.get(False) for _ in range(q.size())]
    assert q.size() == 0
    print(re)

    dq = omniback.ffi.default_queue()
    assert dq is not None

    dq.put([3])
    assert dq.size() == 1, f'dq.size() == {dq.size()},  != 1'
    re = dq.get(False)
    assert re[0] == 3, f'{re}'
    print(f'{dq.size()}')
    with pytest.raises(RuntimeError):
        dq.get(False)
        
    
def test_dict():
    d = omniback.Dict()
    print(d)
    
if __name__ == "__main__":
    test_dict()
    test_queue()