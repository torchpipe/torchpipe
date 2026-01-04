import time


def test_types():
    import omniback
    config = {'instance_num': '2', 'max': '4', 'batching_timeout': 5,
              'backend': 'Identity', 'model': 'yolo11l.onnx'}
    omniback.pipe({"xss": config})


if __name__ == "__main__":
    # time.sleep(5)
    start_time = time.time()
    test_types()
    end_time = time.time()
    print(f"Test completed in {end_time - start_time:.2f} seconds")
