import pytest
import omniback
from typing import Dict, Any

# import toml


from typing import Dict, Any


test_config = {
    "dag_base"
}
def test_configs():
    import time
    # time.sleep(8)
    for x in test_config:
        toml_path = f"config/{x}.toml"

        data = omniback.parse(toml_path)
        print(data)
        omniback.init("Interpreter", {}, omniback.Dict({"config": data}))