import pytest

from typing import List
import hami

class TestPyComponent:
    @pytest.fixture(scope="class")
    def py_instance(self):
        """Fixture to register a PY instance with class scope"""
        instance = PY("test_instance")
        hami.register('py', instance)
        return instance

    @pytest.fixture(scope="class")
    def backend_config(self):
        """Configuration for backend initialization"""
        return {
            "node_name": "test_node",
            "instance_num": "2",  # Changed to integer
            "backend": "Forward[py]"
        }

    @pytest.fixture
    def setup_teardown(self, py_instance, backend_config):
        """Fixture to initialize and clean up backend components"""
        # Initialize backend components
        self.backend_b = hami._C.init(
            "RegisterInstances[BackgroundThread[BackendProxy]]",
            backend_config
        )
        self.backend_a = hami._C.init(
            "RegisterNode[DI[Batching, InstanceDispatcher]]",
            {
                "node_name": backend_config["node_name"],
                "instance_num": backend_config["instance_num"]
            }
        )
        self.list = hami._C.init("List[RegisterInstances[BackgroundThread[BackendProxy]], RegisterNode[DI[Batching, InstanceDispatcher]]]",
                                 {
                                    "node_name": "tmpss",
                                    "instance_num": backend_config["instance_num"],
                                    "backend": backend_config["backend"]
                                })
        yield
        # Add cleanup logic here if required
        # Example: self.backend_a.shutdown()

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
        z=hami._C.get("py")
        z2=hami._C.get("node.test_node")
        assert (z2 and z)
        # print(z)
        
class PY:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def forward(self, inout: List[hami.Dict]):
        """Process data by copying 'data' to 'result'"""
        inout[0]['result'] = inout[0]['data']