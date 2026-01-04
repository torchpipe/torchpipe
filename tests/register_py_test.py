import pytest

from typing import List
import omniback
import time

class TestPyComponent:
    @pytest.fixture(scope="class")
    def py_instance(self):
        """Fixture to register a PY instance with class scope"""

        instance = PY("test_instance")
        omniback.register('py', instance)
        return instance

    @pytest.fixture(scope="function")
    def backend_config(self):
        """Configuration for backend initialization"""
        return {
            "node_name": "test_node"+str(time.time()),
            "instance_num": "2",  # Changed to integer
            "backend": "Forward[py]",
            'node_name_2': "tmpss" + str(time.time()),
        }

    @pytest.fixture
    def setup_teardown(self, py_instance, backend_config):
        """Fixture to initialize and clean up backend components"""
        # Initialize backend components
        import time 
        # time.sleep(15)
        self.backend_b = omniback.init(
            "InstancesRegister[BackgroundThread[BackendProxy]]",
            backend_config
        )
        self.backend_a = omniback.init(
            "Register[IoCV0[SharedInstancesState,InstanceDispatcher,Batching;DI_v0[Batching, InstanceDispatcher]]]",
            {
                "node_name": backend_config["node_name"],
                "instance_num": backend_config["instance_num"]
            }
        )
        self.list = omniback.init("List[InstancesRegister[BackgroundThread[BackendProxy]], Register[IoCV0[SharedInstancesState,InstanceDispatcher,Batching;DI_v0[Batching, InstanceDispatcher]]]]",
                                 {
                                     "node_name": backend_config["node_name_2"],
                                    "instance_num": backend_config["instance_num"],
                                    "backend": backend_config["backend"]
                                })
        yield

    @pytest.mark.parametrize("input_data,expected", [
        ({'data': 3}, 3),
        ({'data': -5}, -5),
        ({'data': 0}, 0),
    ])
    def test_forward_processing(self, setup_teardown, input_data, expected):
        """Test data processing through the backend"""
        test_data = input_data.copy()
        
        # Execute forward pass
        self.backend_a(test_data)
        
        # Verify results
        assert test_data['result'] == expected

    def test_get(self):
        z=omniback.get("py")
        # z2=omniback._C.get("node.test_node.0")
        assert z
        # assert (z2 and z)
        # print(z)
        
class PY:
    def __init__(self, *args, **kwargs) -> None:
        self.data = {"1,2": 3}
        self.g = {"dsfafwed": self.data}

    def forward(self, inout: List[omniback.Dict]):
        """Process data by copying 'data' to 'result'"""
        inout[0]['result'] = inout[0]['data']


 
if __name__ == "__main__":
    pass