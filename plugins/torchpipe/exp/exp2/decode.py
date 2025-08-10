from nvidia import nvimgcodec
import os
import random
import time
import statistics

def get_data(batch_size, img_path='../../tests/encode_jpeg/', gpu_id=0):
    # Collect image paths
    image_paths = []
    for root, dirs, files in os.walk(img_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                image_paths.append(os.path.join(root, file))
    
    print(f"Found {len(image_paths)} images to decode")
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
    
    # Initialize decoder and decode sample images
    dec = nvimgcodec.Decoder(device_id=gpu_id)
    try:
        sample_images = dec.decode(data_list)  # Decode subset for info
        print(f"Successfully decoded {len(sample_images)} sample images")
        for i, img in enumerate(sample_images):
            print(f"\nSample Image {i+1}:")
            print("CUDA Array Interface:", img.__cuda_array_interface__)
            print("Shape:", img.shape)
    except Exception as e:
        print(f"Sample decoding failed: {str(e)}")
    
    batch_data = random.choices(data_list, k=batch_size)
    assert len(batch_data) == batch_size
    return batch_data, dec

def main(batch_size, gpu_id, total=1000, img_path='../../tests/assets/encode_jpeg/'):
    data_list, dec = get_data(batch_size, img_path, gpu_id)
    
    # Warm-up
    _ = dec.decode(data_list)
    
    # Benchmark with time measurement for each iteration
    iteration_times = []
    for _ in range(total):
        start = time.time()
        _ = dec.decode(data_list)
        end = time.time()
        iteration_times.append(end - start)
    
    # Calculate median time per batch
    median_time_per_batch = statistics.median(iteration_times)
    
    # Calculate throughput based on median
    images_per_second = batch_size / median_time_per_batch
    batches_per_second = 1 / median_time_per_batch
    
    print(f"\nBenchmark Results (GPU {gpu_id}):")
    print(f"- Total batches processed: {total}")
    print(f"- Total images decoded: {batch_size * total}")
    print(f"- Median time per batch: {median_time_per_batch * 1000:.4f} ms")
    print(f"- Throughput: {images_per_second:.2f} images/sec")
    print(f"- Batch rate: {batches_per_second:.2f} batches/sec")

if __name__ == '__main__':
    import fire
    fire.Fire(main)