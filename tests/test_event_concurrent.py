"""并发测试：验证 Event 在多线程环境下的线程安全性。

通过多个线程同时操作 Event 对象 / backend 异步路径，
以触发 Event::notify_one/notify_all 的并发调用路径。
"""

import threading

import omniback as om


def test_event_concurrent_wait():
    """多个线程同时对同一个 Event 执行 wait(timeout)，验证无崩溃。"""
    num_threads = 32
    iterations = 100
    errors = []
    barrier = threading.Barrier(num_threads)

    def worker(event):
        try:
            barrier.wait()
            for _ in range(iterations):
                # wait with timeout — will timeout since no notify,
                # but should never crash
                event.wait(5)
        except Exception as e:
            errors.append(e)

    event = om.Event(1)
    threads = [threading.Thread(target=worker, args=(event,)) for _ in range(num_threads)]

    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    assert len(errors) == 0, f"Unexpected errors: {errors}"


def test_backend_concurrent():
    """多个线程同时向 Identity backend 提交任务。

    Backend 内部会通过 evented_forward 路径使用 Event，
    这会触发 notify_one/notify_all 的并发调用。
    """
    num_threads = 16
    iterations = 50
    errors = []
    barrier = threading.Barrier(num_threads)

    backend = om.create("Identity")
    backend.init({}, None)

    def worker(worker_id):
        try:
            barrier.wait()
            for i in range(iterations):
                data = {"data": i}
                backend(data)
                assert data["result"] == i
        except Exception as e:
            errors.append(e)

    threads = []
    for i in range(num_threads):
        t = threading.Thread(target=worker, args=(i,))
        threads.append(t)

    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)

    assert len(errors) == 0, f"Unexpected errors in concurrent backend: {errors}"


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v", "-s"])
