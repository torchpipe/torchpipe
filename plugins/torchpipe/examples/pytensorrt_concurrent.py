#!/usr/bin/env python
"""
High-performance concurrent inference example.

This example demonstrates how to achieve maximum throughput with:
- 10 concurrent requests
- 2 PyTensorrtInferTensor instances (2 CUDA streams)
- 4 batch size per instance
- PySyncTensor for stream synchronization

Architecture:
    10 concurrent requests -> 2 instances x 4 batch_size = 8 concurrent inference
    Each instance uses its own CUDA stream for parallel execution
"""

import torch
import tempfile
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any

from torchpipe.backends.py_tensorrt import PyTensorrtEngine, PyTensorrtInferTensor, ProfileConfig
from torchpipe.backends.cuda_utils import CUDAStreamManager, StreamPool


class SimpleModel(torch.nn.Module):
    """Simple model for benchmarking."""
    
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = torch.nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.pool = torch.nn.MaxPool2d(2, 2)
    
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        return x


class ConcurrentInferenceManager:
    """
    Manager for high-performance concurrent inference.
    
    Uses multiple TensorRT instances with separate CUDA streams
    to maximize GPU utilization.
    """
    
    def __init__(
        self,
        model_path: str,
        num_instances: int = 2,
        batch_size: int = 4,
        use_fp16: bool = True
    ):
        self.model_path = model_path
        self.num_instances = num_instances
        self.batch_size = batch_size
        self.use_fp16 = use_fp16
        
        self.engine = None
        self.backends = []
        self.stream_pool = None
        
        self._init_engine()
        self._init_backends()
        self._init_stream_pool()
    
    def _init_engine(self) -> None:
        """Initialize the TensorRT engine with multiple profiles."""
        print(f"Initializing TensorRT engine with {self.num_instances} profiles...")
        
        self.engine = PyTensorrtEngine(instance_num=self.num_instances)
        
        profiles = []
        for i in range(self.num_instances):
            profiles.append(ProfileConfig(
                min_shapes={'input': (1, 3, 224, 224)},
                opt_shapes={'input': (self.batch_size, 3, 224, 224)},
                max_shapes={'input': (self.batch_size * 2, 3, 224, 224)},
            ))
        
        self.engine.load_from_onnx(
            self.model_path,
            profiles=profiles,
            fp16_mode=self.use_fp16
        )
        
        print(f"Engine created with {self.engine.num_profiles} profiles")
    
    def _init_backends(self) -> None:
        """Initialize backend instances."""
        print(f"Initializing {self.num_instances} backend instances...")
        
        for i in range(self.num_instances):
            backend = PyTensorrtInferTensor()
            backend._engine = self.engine
            backend._context = self.engine.get_or_create_context(i)
            backend._io_info = self.engine.get_io_info(i)
            backend._initialized = True
            backend._input_finish_event = torch.cuda.Event()
            
            self.backends.append(backend)
        
        print(f"Created {len(self.backends)} backend instances")
    
    def _init_stream_pool(self) -> None:
        """Initialize CUDA stream pool."""
        print(f"Initializing stream pool with {self.num_instances} streams...")
        
        self.stream_pool = StreamPool(
            num_streams=self.num_instances,
            high_priority=True
        )
        
        print(f"Stream pool created with {self.stream_pool.num_streams} streams")
    
    def infer_with_stream(self, input_tensor: torch.Tensor, instance_id: int) -> torch.Tensor:
        """
        Run inference with a specific stream.
        
        Args:
            input_tensor: Input tensor
            instance_id: Instance ID (determines which stream to use)
            
        Returns:
            Output tensor
        """
        backend = self.backends[instance_id]
        
        with self.stream_pool.use_stream() as (idx, stream_event):
            if idx != instance_id:
                pass
            
            current_stream = CUDAStreamManager.get_current_stream()
            stream_event.event.record(current_stream)
            
            io_dict = {"data": input_tensor}
            backend.forward([io_dict])
            
            result = io_dict["result"]
            
            stream_event.event.record(current_stream)
        
        return result
    
    def infer_concurrent(self, inputs: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Run concurrent inference on multiple inputs.
        
        Args:
            inputs: List of input tensors
            
        Returns:
            List of output tensors
        """
        results = [None] * len(inputs)
        
        def process_item(idx, inp):
            instance_id = idx % self.num_instances
            return idx, self.infer_with_stream(inp, instance_id)
        
        with ThreadPoolExecutor(max_workers=self.num_instances) as executor:
            futures = [
                executor.submit(process_item, i, inp)
                for i, inp in enumerate(inputs)
            ]
            
            for future in futures:
                idx, result = future.result()
                results[idx] = result
        
        return results


def run_benchmark(
    manager: ConcurrentInferenceManager,
    num_requests: int = 50,
    batch_size: int = 4,
    num_concurrent: int = 10
) -> Dict[str, float]:
    """
    Run benchmark.
    
    Args:
        manager: Inference manager
        num_requests: Number of requests
        batch_size: Batch size
        num_concurrent: Number of concurrent requests
        
    Returns:
        Benchmark results
    """
    print(f"\nBenchmark: {num_requests} requests, {num_concurrent} concurrent, batch_size={batch_size}")
    
    inputs = [
        torch.randn((batch_size, 3, 224, 224), device='cuda')
        for _ in range(num_requests)
    ]
    
    torch.cuda.synchronize()
    start_time = time.time()
    
    results = manager.infer_concurrent(inputs)
    
    torch.cuda.synchronize()
    elapsed_time = time.time() - start_time
    
    latency = elapsed_time / num_requests * 1000
    throughput = num_requests * batch_size / elapsed_time
    
    print(f"  Total time: {elapsed_time:.2f}s")
    print(f"  Latency: {latency:.2f}ms")
    print(f"  Throughput: {throughput:.1f} img/s")
    
    return {
        'total_time': elapsed_time,
        'latency_ms': latency,
        'throughput': throughput,
    }


def main():
    """Main function."""
    print("=" * 60)
    print("High-Performance Concurrent Inference Benchmark")
    print("=" * 60)
    
    # Create model and export to ONNX
    print("\nCreating model...")
    model = SimpleModel()
    model.eval()
    
    tmp_onnx = tempfile.NamedTemporaryFile(suffix='.onnx', delete=False).name
    dummy_input = torch.randn(1, 3, 224, 224)
    torch.onnx.export(
        model,
        dummy_input,
        tmp_onnx,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print(f"ONNX model exported: {tmp_onnx}")
    
    # Configuration
    NUM_INSTANCES = 2      # 2 TensorRT instances (2 CUDA streams)
    BATCH_SIZE = 4         # 4 batch size per instance
    NUM_CONCURRENT = 10    # 10 concurrent requests
    USE_FP16 = True        # Use FP16 for better performance
    
    print(f"\nConfiguration:")
    print(f"  - Instances: {NUM_INSTANCES}")
    print(f"  - Batch size: {BATCH_SIZE}")
    print(f"  - Concurrent requests: {NUM_CONCURRENT}")
    print(f"  - FP16 mode: {USE_FP16}")
    
    # Create manager
    manager = ConcurrentInferenceManager(
        model_path=tmp_onnx,
        num_instances=NUM_INSTANCES,
        batch_size=BATCH_SIZE,
        use_fp16=USE_FP16
    )
    
    try:
        # Warmup
        print("\nWarming up...")
        for _ in range(10):
            inp = torch.randn((BATCH_SIZE, 3, 224, 224), device='cuda')
            manager.infer_with_stream(inp, 0)
        torch.cuda.synchronize()
        print("Warmup complete")
        
        # Run benchmark
        print("\n" + "=" * 60)
        print("Running Benchmark")
        print("=" * 60)
        
        results = run_benchmark(
            manager,
            num_requests=50,
            batch_size=BATCH_SIZE,
            num_concurrent=NUM_CONCURRENT
        )
        
        # Summary
        print("\n" + "=" * 60)
        print("Summary")
        print("=" * 60)
        print(f"Configuration: {NUM_INSTANCES} instances x {BATCH_SIZE} batch = {NUM_INSTANCES * BATCH_SIZE} concurrent images")
        print(f"Throughput: {results['throughput']:.1f} img/s")
        print(f"Latency: {results['latency_ms']:.2f}ms")
        
    finally:
        # Cleanup
        os.remove(tmp_onnx)
        print(f"\nCleaned up: {tmp_onnx}")
    
    print("\n✓ Benchmark complete!")


if __name__ == "__main__":
    main()
