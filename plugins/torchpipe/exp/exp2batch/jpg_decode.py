from nvidia import nvimgcodec
import os
import random
import time
import statistics
import torch

def get_data(batch_size, img_path='../../tests/encode_jpeg/', gpu_id=0):
    # Collect image paths
    image_paths = []
    for root, dirs, files in os.walk(img_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                image_paths.append(os.path.join(root, file))
    
    print(f"Found {len(image_paths)} images")
    if len(image_paths) == 0:
        print("No images found. Exiting.")
        exit(1)

    # Read binary data
    data_list = []
    for p in image_paths:
        try:
            with open(p, 'rb') as in_file:
                data_list.append(in_file.read())
        except Exception as e:
            print(f"Error reading file {p}: {str(e)}")
    
    if len(data_list) == 0:
        print("No valid images read. Exiting.")
        exit(1)

    # Initialize decoder
    dec = nvimgcodec.Decoder(device_id=gpu_id)
    return data_list, dec

def main(batch_size, gpu_id, total=4000, img_path='../../tests/assets/encode_jpeg/'):
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        assert str(gpu_id) == os.environ['CUDA_VISIBLE_DEVICES'] or gpu_id == 0
        gpu_id = 0
    else:
        assert gpu_id != 0
    torch.cuda.set_device(gpu_id)
    all_data, dec = get_data(batch_size, img_path, gpu_id)
    
    # Select random batch (consistent across runs)
    batch_data = random.choices(all_data, k=batch_size)
    
    print(f'Warm-up started (batch_size={batch_size})')
    # Warm-up with GPU synchronization
    for _ in range(5):
        _ = dec.decode(batch_data)
    torch.cuda.synchronize()
    print('Warm-up finished')
    
    # Benchmark with CUDA events
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    iteration_times = []
    
    for _ in range(total):
        start_event.record()
        _ = dec.decode(batch_data)
        end_event.record()
        torch.cuda.current_stream().synchronize()  # Ensure measurement completes
        elapsed_ms = start_event.elapsed_time(end_event)
        iteration_times.append(elapsed_ms / 1000.0)  # Convert to seconds
    
    # Calculate statistics
    median_time_per_batch = statistics.median(iteration_times)
    images_per_second = batch_size / median_time_per_batch
    batches_per_second = 1 / median_time_per_batch
    
    print(f"\nBenchmark Results (GPU {gpu_id}):")
    print(f"- Total batches processed: {total}")
    print(f"- Total images decoded: {batch_size * total}")
    print(f"- Median time per batch: {median_time_per_batch * 1000:.4f} ms")
    print(f"- Throughput: {batches_per_second:.2f} qps")

if __name__ == '__main__':
    import fire
    fire.Fire(main)