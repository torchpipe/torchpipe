
import omniback
import pytest
def test_dup():
    config1 = {"2212": {"backend": "Identity"}}
    config2 = {"2212": {"backend": "Identity"}}
    a=omniback.pipe(config1)
    with pytest.raises(RuntimeError):
        b = omniback.pipe(config2)
    
if __name__ == '__main__':
    test_dup()