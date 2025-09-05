
import hami

def test_dup():
    config1 = {"a": {"backend":"Identity"}}
    config2 = {"a": {"backend": "Identity"}}
    a=hami.pipe(config1)
    b = hami.pipe(config2)
    
if __name__ == '__main__':
    test_dup()