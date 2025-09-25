
import hami
import pytest
def test_dup():
    config1 = {"2212": {"backend": "Identity"}}
    config2 = {"2212": {"backend": "Identity"}}
    a=hami.pipe(config1)
    with pytest.raises(RuntimeError):
        b = hami.pipe(config2)
    
if __name__ == '__main__':
    test_dup()