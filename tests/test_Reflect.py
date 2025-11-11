import omniback
# import time
# time.sleep(10)
import pytest

def test_not_exist():
    with pytest.raises(RuntimeError):
        omniback.init("Reflect[post_processor,Identity]", {"post_processor": "NotExist"})