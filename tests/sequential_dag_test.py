import pytest
import omniback
from typing import Dict, Any

class BackendManager:
    BASE_CONFIG = {
        "instance_num": "2",
        "backend": "Identity"
    }
    
    BACKEND_TEMPLATE = "List[InstancesRegister[BackgroundThread[Reflect]],Register[IoCV0[SharedInstancesState,InstanceDispatcher,Batching;DI_v0[Batching, InstanceDispatcher]]]]"

    def __init__(self):
        self.backends = {}
        self._init_backends()

    def _init_backends(self):
        backend_configs = [
            {"node_name": "node_a", "backend": "Identity"},
            {"node_name": "node_b", "backend": "Pow"},
            {"node_name": "node_c", "backend": "Pow"},
            {"node_name": "node_d", "backend": "Pow"}
        ]
        
        for config in backend_configs:
            node_name = config["node_name"]
            full_config = {**self.BASE_CONFIG, **config}
            # print(full_config)
            self.backends[node_name] = omniback.init(self.BACKEND_TEMPLATE, full_config)

def create_dag_config() -> Dict[str, Any]:
    return {
        "node_c": {"map": "node_a[result:data]"},
        "node_b": {"next": "node_c"},
        "node_a": {"next": "node_b"}
    }

@pytest.fixture
def backend_manager():
    return BackendManager()

@pytest.fixture
def dag_model():
    dag_config = create_dag_config()
    return omniback.init("DagDispatcher", {}, omniback.Dict({"config": dag_config}))

def test_dag(backend_manager, dag_model):
    input_data = {"data": 2, "node_name": "node_a"}
    dag_model(input_data)
    assert input_data["result"] == 4, "Expected result to be 4"

if __name__ == "__main__":
    pytest.main([__file__])