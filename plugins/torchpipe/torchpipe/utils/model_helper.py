# Model_helper.py
import importlib
import os
import subprocess
import urllib.request
from io import BytesIO
from typing import Callable, Tuple, Optional

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import tempfile
import logging
logging.basicConfig(level=logging.INFO)
import importlib
import subprocess




TEST_IMAGES = [
        # PyTorch official examples
        ("imagenet_dog", "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg"),
        
        # TensorFlow example images
        ("tf_grace_hopper", "https://storage.googleapis.com/download.tensorflow.org/example_images/grace_hopper.jpg"),

        # Sample images from public domain sources
        ("pexels_nature", "https://images.pexels.com/photos/3608263/pexels-photo-3608263.jpeg"),
        
        # Small images
        ("small_icon1", "https://www.google.com/favicon.ico"),  # 16x16 favicon
        ("small_icon2", "https://github.com/favicon.ico"),  # 32x32 favicon
        ("small_icon3", "https://raw.githubusercontent.com/pytorch/pytorch/master/docs/source/_static/img/pytorch-logo-flame.png"),  # Small PyTorch logo
        ("tiny_sample", "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/smarties.png"),  # Small test image from OpenCV
        ("cat", "http://images.cocodataset.org/val2017/000000039769.jpg")
    ]

def import_or_install_package(package_name: str) -> None:
    """Dynamically import or install missing packages.
    
    Args:
        package_name: Name of the package to install/import
    """
    try:
        module = importlib.import_module(package_name)
    except ImportError:
        print(f"Installing required package: {package_name}")
        subprocess.run(["pip", "install", package_name], check=True)
        module = importlib.import_module(package_name)
    return module


def export_x3hw(model: nn.Module, onnx_path: str, input_h: int, input_w: int) -> str:
    """Export model to optimized ONNX format with input shape [-1, 3, H, W].
    
    Args:
        model: PyTorch model to export
        input_h: Input height
        input_w: Input width

    Returns:
        Path to the exported ONNX model
    """
    # Ensure dependencies are installed
    for package in ["onnxsim", "onnx"]:
        import_or_install_package(package)
    
    # Import required modules after installation
    import onnx
    from onnxsim import simplify
    
    # Generate dummy input
    dummy_input = torch.randn(1, 3, input_h, input_w)
    tmp_onnx_path = tempfile.mktemp()
    
    
    
    
    # Export initial ONNX model
    torch.onnx.export(
        model,
        dummy_input,
        tmp_onnx_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        opset_version=13
    )
    
    # Optimize with ONNX-Simplifier
    onnx_model = onnx.load(tmp_onnx_path)
    os.remove(tmp_onnx_path)
    model_simp, check = simplify(onnx_model)
    assert check, "Simplified ONNX model could not be validated"
    onnx.save(model_simp, onnx_path)
    print(f"Optimized ONNX model saved to: {onnx_path}")
    return onnx_path


def get_classification_model(name: str, input_h: int, input_w: int) -> Tuple[nn.Module, Callable]:
    """Get TIMM classification model with preprocessing function.
    
    Args:
        name: Model architecture name
        input_h: Input image height
        input_w: Input image width
    
    Returns:
        Tuple containing:
        - Initialized classification model
        - Preprocessing function that accepts bytes and returns torch.Tensor
    """
    import_or_install_package("timm")
    from timm import create_model
    
    model = create_model(
        name,
        pretrained=True,
        num_classes=1000
    )
    model.eval()
    
    # 创建包含字节解码的预处理流程
    preprocess = transforms.Compose([
        # 添加字节解码步骤
        transforms.Lambda(lambda x: Image.open(BytesIO(x)).convert('RGB')),
        transforms.Resize((input_h, input_w)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return model, preprocess


def download_image(url: str) -> Optional[bytes]:
    """Download image from url.
    
    Args:
        url: Image URL
    
    Returns:
        raw image data or None if failed
    """
    try:
        with urllib.request.urlopen(url) as response:
            img_data = response.read()
        # Verify the image can be opened
        Image.open(BytesIO(img_data)).verify()
        return img_data
    except Exception as e:
        print(f"Image download or processing failed: {str(e)}")
        return None


def download_and_preprocess_image(url: str, preprocess: Callable) -> Optional[torch.Tensor]:
    """Download an image, preprocess it, and return a tensor.
    
    Args:
        url: URL of the image
        preprocess: Preprocessing function to apply
    
    Returns:
        Preprocessed tensor or None if an error occurs
    """
    img_bytes = download_image(url)
    if img_bytes is None:
        return None
    
    try:
        image = Image.open(BytesIO(img_bytes)).convert('RGB')
        input_tensor = preprocess(image)
        return input_tensor.unsqueeze(0)  # Add batch dimension
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

class OnnxModel:
        
    def __init__(self, onnx_path, preprocessor):
        self.preprocessor = preprocessor
        
        onnxruntime = import_or_install_package("onnxruntime")
        self.session = onnxruntime.InferenceSession(onnx_path)
        self.input_name = self.session.get_inputs()[0].name
    
    def __call__(self, img):
        input_data = self.preprocessor(img).unsqueeze(0).numpy()
        result = self.session.run(None, {self.input_name: input_data})
        return result[0]

        
def report_classification(all_result):
    """
    Report comparison statistics between PyTorch and ONNX model results for classification models.
    
    Args:
        all_result: Dictionary mapping data_id to tuples of (pytorch_result, onnx_result)
    
    Statistics reported:
        - Total number of samples compared
        - Number of samples with similar tensor values (using allclose)
        - Number of samples with the same predicted class (argmax)
        - Percentage of value similarity and class prediction matching
        - Samples with largest numerical differences
        - Samples with class prediction changes
    """
    total_count = len(all_result)
    similar_count = 0
    same_class_count = 0
    
    # Variables to track largest differences
    max_abs_diff = 0.0
    max_diff_data_id = None
    max_diff_classes = None
    
    # Variables to track largest probability difference for top class
    max_top_class_diff = 0.0
    max_top_class_diff_id = None
    
    # Variables to track mismatched predictions
    class_mismatches = []
    
    # Iterate through all results to compare
    for data_id, (result, onnx_result) in all_result.items():
        # Check if tensor values are similar
        is_similar = torch.allclose(result, onnx_result, atol=1e-3)
        if is_similar:
            similar_count += 1
            
        # Calculate absolute difference between tensors
        abs_diff = torch.abs(result - onnx_result)
        current_max_diff = torch.max(abs_diff).item()
        
        # Update max difference if current difference is larger
        if current_max_diff > max_abs_diff:
            max_abs_diff = current_max_diff
            max_diff_data_id = data_id
            
        # Check if predicted class (argmax) is the same
        pytorch_class = torch.argmax(result).item()
        onnx_class = torch.argmax(onnx_result).item()
        is_same_class = pytorch_class == onnx_class
        
        if is_same_class:
            same_class_count += 1
        else:
            # Store information about mismatched predictions
            pytorch_prob = result[0, pytorch_class].item() if result.dim() > 1 else result[pytorch_class].item()
            onnx_prob = onnx_result[0, onnx_class].item() if onnx_result.dim() > 1 else onnx_result[onnx_class].item()
            mismatched_info = {
                'data_id': data_id,
                'pytorch_class': pytorch_class,
                'onnx_class': onnx_class,
                'pytorch_prob': pytorch_prob,
                'onnx_prob': onnx_prob,
                'abs_diff': current_max_diff
            }
            class_mismatches.append(mismatched_info)
            
            # Update max class difference info
            if max_diff_classes is None or current_max_diff > max_abs_diff:
                max_diff_classes = (pytorch_class, onnx_class)
        
        # Calculate difference in probability for top class
        pytorch_top_prob = result[0, pytorch_class].item() if result.dim() > 1 else result[pytorch_class].item()
        onnx_top_prob_for_pytorch_class = onnx_result[0, pytorch_class].item() if onnx_result.dim() > 1 else onnx_result[pytorch_class].item()
        top_class_diff = abs(pytorch_top_prob - onnx_top_prob_for_pytorch_class)
        
        if top_class_diff > max_top_class_diff:
            max_top_class_diff = top_class_diff
            max_top_class_diff_id = data_id
            
        # Print individual result details
        if current_max_diff > 1e-6:
            print(f"Data ID: {data_id}")
            print(f"PyTorch Result: {result.shape}")
            print(f"ONNX Result: {onnx_result.shape}")
            print(f"Max Difference: {current_max_diff:.6f}")
            print("Value similarity: " + ("Similar" if is_similar else "Different"))
            print("Class prediction: " + ("Same" if is_same_class else "Different"))
            print("-" * 50)
    
    # Sort mismatches by the absolute difference (largest first)
    class_mismatches.sort(key=lambda x: x['abs_diff'], reverse=True)
    
    # Calculate and print overall statistics
    value_similarity_pct = (similar_count / total_count) * 100 if total_count > 0 else 0
    class_match_pct = (same_class_count / total_count) * 100 if total_count > 0 else 0
    
    print("\n===== Overall Statistics =====")
    print(f"Total samples: {total_count}")
    print(f"Samples with similar values: {similar_count}/{total_count} ({value_similarity_pct:.2f}%)")
    print(f"Samples with same class prediction: {same_class_count}/{total_count} ({class_match_pct:.2f}%)")
    
    # Report largest differences
    print("\n===== Largest Differences =====")
    print(f"Sample with largest tensor difference: Data ID {max_diff_data_id}, Max diff: {max_abs_diff:.6f}")
    print(f"Sample with largest top class probability difference: Data ID {max_top_class_diff_id}, Diff: {max_top_class_diff:.6f}")
    
    # Report class prediction mismatches (up to 5)
    if class_mismatches:
        print("\n===== Top Class Prediction Mismatches =====")
        for i, mismatch in enumerate(class_mismatches[:5]):
            print(f"Mismatch {i+1}: Data ID {mismatch['data_id']}")
            print(f"  PyTorch: Class {mismatch['pytorch_class']} with prob {mismatch['pytorch_prob']:.6f}")
            print(f"  ONNX: Class {mismatch['onnx_class']} with prob {mismatch['onnx_prob']:.6f}")
            print(f"  Max difference in tensor: {mismatch['abs_diff']:.6f}")
    
    print("==============================")
    
class TestImageDataset:
    """
    A dataset class for testing images from common deep learning sources.
    
    This dataset provides a PyTorch-like interface for iterating through test images,
    with support for resetting and reusing the iterator.
    
    Args:
        image_sources (list, optional): List of (image_id, url) tuples. 
            If None, uses predefined TEST_IMAGES.
        timeout (float, optional): Request timeout in seconds. Default: 10.0
        max_retries (int, optional): Maximum number of download retries. Default: 2
        logger (Logger, optional): Custom logger. If None, creates a default one.
        
    Examples::
        >>> dataset = TestImageDataset()
        >>> for image_id, image_bytes in dataset:
        ...     process_image(image_id, image_bytes)
        ...
        >>> # Reset and use again
        >>> dataset.reset()
        >>> for image_id, image_bytes in dataset:
        ...     process_image(image_id, image_bytes)
    """
    
    def __init__(self, image_sources=None, timeout=10.0, max_retries=2, logger=None):
        import logging
        
        self.image_sources = image_sources if image_sources is not None else TEST_IMAGES
        self.timeout = timeout
        self.max_retries = max_retries
        self.logger = logger or logging.getLogger("TestImageDataset")
        self._iterator = None
        self.failed_image_ids = []
    
    def __iter__(self):
        """Return the dataset iterator."""
        self._iterator = self._create_iterator()
        return self
    
    def __next__(self):
        """Get next (image_id, image_bytes) pair."""
        if self._iterator is None:
            self._iterator = self._create_iterator()
        
        try:
            return next(self._iterator)
        except StopIteration:
            self._log_failures()
            raise
    
    def __len__(self):
        """Return the number of images in the dataset."""
        return len(self.image_sources)
    
    def reset(self):
        """Reset the iterator to start from the beginning."""
        self._iterator = None
        self.failed_image_ids = []
        return self
    
    def _create_iterator(self):
        """Create a fresh iterator for the dataset."""
        import requests
        from io import BytesIO
        from urllib.parse import urlparse
        
        self.failed_image_ids = []
        
        for image_id, url in self.image_sources:
            parsed_url = urlparse(url)
            filename = parsed_url.path.split('/')[-1]
            full_id = image_id
            
            # Try downloading with retries
            image_bytes = None
            for attempt in range(self.max_retries + 1):
                try:
                    response = requests.get(url, timeout=self.timeout)
                    response.raise_for_status()
                    image_bytes = BytesIO(response.content).getvalue()
                    break
                except Exception as e:
                    if attempt < self.max_retries:
                        self.logger.debug(f"Retry {attempt+1}/{self.max_retries} for {full_id}: {str(e)}")
                    else:
                        self.logger.warning(f"Failed to download {full_id}: {str(e)}")
                        self.failed_image_ids.append(full_id)
            
            yield full_id, image_bytes
    
    def _log_failures(self):
        """Log information about failed downloads."""
        if self.failed_image_ids:
            self.logger.error(f"Failed to download {len(self.failed_image_ids)} images:")
            for img_id in self.failed_image_ids:
                self.logger.error(f"  - {img_id}")

    @property
    def failed_count(self):
        """Return the number of images that failed to download."""
        return len(self.failed_image_ids)
    
       
def test_onnx():
    model, preprocessor = get_classification_model("resnet50", 224, 224)
    
    onnx_path =  f"{model.__class__.__name__}.onnx"
    export_x3hw(model,onnx_path, 224, 224)
    
    onnx_model = OnnxModel(onnx_path, preprocessor)

    all_result = {}
    dataset = TestImageDataset()
    for data_id, data in TestImageDataset():
        if data is not None:
            preprocessed = preprocessor(data).unsqueeze(0)
            with torch.no_grad():
                result = torch.nn.functional.softmax(model(preprocessed), dim=-1)
            onnx_result = torch.nn.functional.softmax(torch.from_numpy(onnx_model(data)[0]), dim=-1)
            all_result[data_id] = (result, onnx_result)
    report_classification(all_result)
    
class ClassifyModelTester:
    def __init__(self, model_name, onnx_path, h = 224, w = 224):

        self.model_name = model_name
        self.model, self.preprocessor = get_classification_model(self.model_name, h, w)
    
        export_x3hw(self.model, onnx_path, h, w)
        
    def test(self, callable_func):
        all_result = {}
        dataset = TestImageDataset()
        for data_id, data in dataset:
            if data is not None:
                preprocessed = self.preprocessor(data).unsqueeze(0)
                with torch.no_grad():
                    result = torch.nn.functional.softmax(self.model(preprocessed), dim=-1)
                    extra_result = torch.nn.functional.softmax(callable_func(data), dim=-1)
                all_result[data_id] = (result, extra_result)
        report_classification(all_result)
        return all_result
        
if __name__ == "__main__":
    test_onnx()
    