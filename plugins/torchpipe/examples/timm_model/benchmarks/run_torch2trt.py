import fire
import time
from torch2trt import torch2trt,TRTModule
import timm
import os
import argparse
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import torch.nn as nn
import torch
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import queue
from io import BytesIO

class TimmTRTModel:
    def __init__(
        self,
        model_name: str = "eva02_base_patch14_448.mim_in22k_ft_in22k_in1k",
        precision: str = "fp16",
        max_batch_size: int = 1,
        use_cache: bool = True,
        cache_path: str = "./.cache/torch2trt"
    ):
        """
        Initialize the TensorRT model wrapper.
        
        Args:
            model_name: Name of the timm model to load
            precision: TensorRT precision ("fp32" or "fp16")
            max_batch_size: Maximum batch size for TRT engine
            use_cache: Whether to cache the TRT engine
            cache_path: Directory to store cached engines
        """
        self.model_name = model_name
        self.precision = precision
        self.max_batch_size = max_batch_size
        self.use_cache = use_cache
        self.cache_path = cache_path

        # Set device
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        # Lock for thread-safe inference
        self.inference_lock = threading.Lock()

        # Load and convert model
        self.model_trt = None
        self._load_and_convert_model()
        self.preprocess = transforms.Compose([
            transforms.Resize(
                (self.input_size[1], self.input_size[2])),  # todo
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def _load_and_convert_model(self):
        """Load the timm model and convert to TensorRT."""
        print(f"Loading model: {self.model_name}")
    
        # Determine precision flag
        fp16_mode = (self.precision.lower() == "fp16")

        # Check if cached engine exists
        if self.use_cache:
            os.makedirs(self.cache_path, exist_ok=True)
            cache_filename = os.path.join(
                self.cache_path,
                f"{self.model_name}_{self.precision}.trt"
            )

            if os.path.exists(cache_filename):
                print(f"Loading cached TRT engine from: {cache_filename}")

                self.model_trt = TRTModule()
                self.model_trt.load_state_dict(torch.load(
                    cache_filename, map_location=self.device))
                print("Cached TRT engine loaded successfully.")

        model = timm.create_model(
                self.model_name,
                pretrained=False,
                num_classes=3  # Remove classifier head for feature extraction
            )
        self.input_size = model.default_cfg.get('input_size', (3, 224, 224))
            # Create example input for conversion
        example_input = torch.randn(1, *self.input_size).to(self.device)
            
        if self.model_trt is None:
            print(
                f"Converting model to TensorRT with precision: {self.precision}")
            # Load the model from timm
            model = timm.create_model(
                self.model_name,
                pretrained=True,
                num_classes=3  # Remove classifier head for feature extraction
            )
            model.eval()
            model = model.to(self.device)
            
            self.model_trt = torch2trt(
                model,
                [example_input],
                fp16_mode=fp16_mode,
                max_batch_size=self.max_batch_size,
                max_workspace_size=1 << 30,  # 1GB
                use_onnx=True,
            )
            # save cached engine
            if self.use_cache:
                print(f"Caching TRT engine to: {cache_filename}")
                torch.save(self.model_trt.state_dict(), cache_filename)
                print("TRT engine cached successfully.")

        # Warmup
        print("Warming up...")
        with torch.no_grad():
            for _ in range(3):
                _ = self.model_trt(example_input)

    def preprocess_image(self, image_path: str):
        """Load and preprocess image for the model."""
        if isinstance(image_path, str):
            image = Image.open(image_path).convert('RGB')
        elif isinstance(image_path, bytes):
            image = Image.open(BytesIO(image_path)).convert('RGB')

        image_tensor = self.preprocess(image)
        image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension
        return image_tensor

    def inference(self, input_tensor: torch.Tensor):
        """
        Run inference on the input tensor.
        
        Args:
            input_tensor: Preprocessed input tensor
            batch_size: Batch size for inference
            
        Returns:
            Model output tensor
        """
        
        # Thread-safe inference using lock
        with self.inference_lock:
            with torch.no_grad():
                input_tensor = input_tensor.to(self.device)
                torch.cuda.current_stream().synchronize()  # Ensure previous GPU tasks are done
                output = self.model_trt(input_tensor)
                output = output.cpu()
                torch.cuda.current_stream().synchronize()

        return output
 

def get_client(image_path):
    data = open(image_path, 'rb').read()
    # from run_torch2trt import TimmTRTModel
    local_model = TimmTRTModel(
        model_name="eva02_base_patch14_448.mim_in22k_ft_in22k_in1k")

    def forward_func(ids):
        preprocessed = local_model.preprocess_image(data)
        re = local_model.inference(preprocessed)
        return re

    return forward_func

def main(
    model_name: str = "eva02_base_patch14_448.mim_in22k_ft_in22k_in1k",
    image_path: str = "../../tests/assets/encode_jpeg/grace_hopper_517x606.jpg",
    precision: str = "fp16",  # "fp32" or "fp16"
    max_batch_size: int = 1,
    use_cache: bool = True,
    cache_path: str = "./.cache/torch2trt",
):
    """
    Convert and run inference on timm model using torch2trt.
    
    Args:
        model_name: Name of the timm model to load
        image_path: Path to input image
        batch_size: Batch size for inference
        precision: TensorRT precision ("fp32" or "fp16")
        max_batch_size: Maximum batch size for TRT engine
        use_cache: Whether to cache the TRT engine
        cache_path: Directory to store cached engines
        benchmark: Whether to run benchmarking
        concurrent: Whether to run concurrent inference
        num_concurrent: Number of concurrent inference requests
        num_benchmark_runs: Number of runs for benchmarking
    """
    # Initialize model
    trt_model = TimmTRTModel(
        model_name=model_name,
        precision=precision,
        max_batch_size=max_batch_size,
        use_cache=use_cache,
        cache_path=cache_path
    )

    # Preprocess image
    print(f"Loading image: {image_path}")
    input_tensor = trt_model.preprocess_image(image_path)

    print("Running single inference...")
    output = trt_model.inference(input_tensor)
    print(f"Output: {output}")


if __name__ == "__main__":
    fire.Fire(main)
