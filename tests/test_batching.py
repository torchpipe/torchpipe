import omniback
import pytest
import time
import threading

def test_batching_timeout_zero():
    """Test Batching backend with timeout=0 (should not hang or crash)."""
    # Create a pipeline that uses Batching
    # Identity backend with max=4 will trigger Batching if instance_num > 1
    config = {
        "test_node": {
            "backend": "Identities",
            "instance_num": 2,
            "batching_timeout": 0,
            "max": 4
        }
    }

    # pipe() creates the pipeline
    # The structure should be:
    # Batching -> InstanceDispatcher -> Identities
    model = omniback.pipe(config)

    # Send some requests
    # Since timeout=0, it should process immediately or wait for max batch?
    # If timeout=0, logic says:
    # 1. Wait for first item (blocking)
    # 2. Once item arrives, if timeout=0, try to get more up to max, then send.

    # Send one request (less than max)
    inp = omniback.Dict({'data': 1})
    model(inp)
    assert inp['result'] == 1

    # Send multiple requests
    inps = [omniback.Dict({'data': i}) for i in range(10)]
    # Use threads to simulate concurrent requests

    def call_model(d):
        model(d)

    threads = [threading.Thread(target=call_model, args=(d,)) for d in inps]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    for i, d in enumerate(inps):
        assert d['result'] == i

if __name__ == "__main__":
    test_batching_timeout_zero()
