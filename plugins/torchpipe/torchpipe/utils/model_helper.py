# Model_helper.py
import importlib
import os
import subprocess
import urllib.request
from io import BytesIO
from typing import Callable, Tuple, Optional
import numpy as np
from typing import List, Union, Callable, Tuple, Any
try:
    import cv2
except ImportError:
    print("OpenCV not found. Please install it to use this feature.")
    
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import tempfile
import logging
import importlib
import subprocess




# TEST_IMAGES = [
#         # PyTorch official examples
#         ("imagenet_dog", "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg"),
        
#         # TensorFlow example images
#         ("tf_grace_hopper", "https://storage.googleapis.com/download.tensorflow.org/example_images/grace_hopper.jpg"),

#         # Sample images from public domain sources
#         ("pexels_nature", "https://images.pexels.com/photos/3608263/pexels-photo-3608263.jpeg"),
        
#         # # Small images
#         ("small_icon1", "https://www.google.com/favicon.ico"),  # 16x16 favicon
#         ("small_icon2", "https://github.com/favicon.ico"),  # 32x32 favicon
#         ("small_icon3", "https://raw.githubusercontent.com/pytorch/pytorch/master/docs/source/_static/img/pytorch-logo-flame.png"),  # Small PyTorch logo
#         ("tiny_sample", "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/smarties.png"),  # Small test image from OpenCV
#         ("cat", "http://images.cocodataset.org/val2017/000000039769.jpg")
#     ]
IMAGENET_IMAGES = [
    ('bird', 'https://modelscope.oss-cn-beijing.aliyuncs.com/test/images/bird.JPEG', 0),
    ('African Elephant', 'https://obs-9be7.obs.cn-east-2.myhuaweicloud.com/models/aclsample/dog1_1024_683.jpg', 101),
    ("Golden Retriever", "https://obs-9be7.obs.cn-east-2.myhuaweicloud.com/models/aclsample/dog2_1024_683.jpg", 207),
            ("cat", "http://images.cocodataset.org/val2017/000000039769.jpg", 0),
            ("imagenet_dog", "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg", 0),
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

def preprocess_with_cv2(image_bytes, input_h = 224, input_w = 224):
    nparr = np.frombuffer(image_bytes, np.uint8) # 转换为numpy数组
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # 转换为 OpenCV 的 BGR 格式

    image = cv2.resize(image, (input_w, input_h))

    image = image.transpose((2, 0, 1))  # HWC to CHW
    image = image.astype(np.float32) / 255.0  # 归一化到 [0, 1]

    mean = np.array([0.485, 0.456, 0.406], dtype = np.float32).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225], dtype = np.float32).reshape(3, 1, 1)
    image = (image - mean) / std

    return torch.from_numpy(image)

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


def get_mini_imagenet():
    from modelscope.msdatasets import MsDataset
    from modelscope.utils.constant import DownloadMode


    # {'image:FILE': '*10.JPEG', 'category': 0}

    ms_val_dataset = MsDataset.load(
                'mini_imagenet100', namespace='tany0699',
                subset_name='default', split='validation',
                download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS) # 加载验证集
    # print(next(iter(ms_val_dataset)))
    return ms_val_dataset



def align_labels(y_true, y_pred, num_class):
    from statistics import median
    y_len = len(y_true)
    assert len(y_pred) == y_len
    total = {}
    for i in range(y_len):
        if y_true[i] not in total:
            total[y_true[i]] = []
        total[y_true[i]].append(y_pred[i])
    assert (num_class == len(total))
    map_y = {}
    for k, v in total.items():
        assert len(v) > 0
        mid = int(median(v))
        assert mid not in map_y.keys()
        map_y[mid] = k

    for y in y_pred:
        if y not in map_y:
            map_y[y] = num_class
    return map_y


def evaluate_classification(
    y_true, 
    y_pred, 
    class_names=None,
    verbose: bool = True
) -> dict:
    """
    Comprehensive classification evaluation.

    Args:
        y_true: Array-like of ground truth labels
        y_pred: Array-like of predicted labels
        class_names: Optional list of class names for reporting
        verbose: Enable progress visualization (default: True)

    Returns:
        Dictionary containing:
        - Metrics (accuracy/precision/recall/f1)
        - Classification report (if class_names provided)
    """
    import numpy as np
    from sklearn.metrics import (
        accuracy_score, precision_score,
        recall_score, f1_score,
        confusion_matrix, classification_report
    )
    from scipy.optimize import linear_sum_assignment
    from tqdm import tqdm
    import warnings

    # Input validation
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    
    if len(y_true) != len(y_pred):
        raise ValueError("Input lengths must match")

    # Label alignment system
    mapping = None
    
    # Metric calculation with safe type handling
    metrics = {}
    steps = [
        ('accuracy', lambda t, p: accuracy_score(t, p)),
        ('precision', lambda t, p: precision_score(t, p, average='macro', zero_division=0)),
        ('recall', lambda t, p: recall_score(t, p, average='macro', zero_division=0)),
        ('f1', lambda t, p: f1_score(t, p, average='macro', zero_division=0)),
        ('confusion_matrix', lambda t, p: confusion_matrix(t, p))
    ]

    if verbose:
        steps = tqdm(steps, desc="Computing metrics", disable=not verbose)

    for name, func in steps:
        try:
            result = func(y_true.astype(int), y_pred.astype(int))
            metrics[name] = result
        except Exception as e:
            if verbose:
                print(f"Error calculating {name}: {str(e)}")
            metrics[name] = None

    # Classification report handling
    if class_names is not None:
        try:
            metrics['classification_report'] = classification_report(
                y_true, y_pred,
                target_names=class_names,
                output_dict=True
            )
        except Exception as e:
            if verbose:
                print(f"Error generating report: {str(e)}")


    # Verbose output formatting
    if verbose:
        print("\n\033[1mEvaluation Summary:\033[0m")
        print(f"- Accuracy:  {metrics.get('accuracy', 'N/A'):.4f}")
        print(f"- Precision: {metrics.get('precision', 'N/A'):.4f}")
        print(f"- Recall:    {metrics.get('recall', 'N/A'):.4f}")
        print(f"- F1 Score:  {metrics.get('f1', 'N/A'):.4f}")
        
        if 'classification_report' in metrics:
            print("\n\033[1mClassification Report:\033[0m")
            print(classification_report(
                y_true, y_pred,
                target_names=class_names,
                output_dict=False
            ))

    return metrics
    
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
        is_similar = torch.allclose(result, onnx_result, atol=1e-1, rtol=0.2)
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
            print(f"PyTorch Result: {result.shape} max = {result.max().item()}")
            print(f"user's Result: {onnx_result.shape} max = {onnx_result.max().item()}")
            print(f"Max Difference: {current_max_diff:.6f}")
            print("Value similarity: " + ("Similar" if is_similar else "Different"))
            print("Class prediction: " + ("Same" if is_same_class else "Different") + f" ({pytorch_class} vs {onnx_class})")
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
    print(f"Sample with largest tensor difference: Data ID: {max_diff_data_id}, Max diff: {max_abs_diff:.6f}")
    print(f"Sample with largest top class probability difference: Data ID: {max_top_class_diff_id}, Diff: {max_top_class_diff:.6f}")
    
    # Report class prediction mismatches (up to 5)
    if class_mismatches:
        print("\n===== Top Class Prediction Mismatches =====")
        for i, mismatch in enumerate(class_mismatches[:5]):
            print(f"Mismatch {i+1}: Data ID {mismatch['data_id']}")
            print(f"  PyTorch: Class {mismatch['pytorch_class']} with prob {mismatch['pytorch_prob']:.6f}")
            print(f"  ONNX: Class {mismatch['onnx_class']} with prob {mismatch['onnx_prob']:.6f}")
            print(f"  Max difference in tensor: {mismatch['abs_diff']:.6f}")
    
    print("==============================")
    assert similar_count == total_count, "All samples should have similar values"
    assert same_class_count == total_count, "All samples should have the same class prediction"
    
class TestImageDataset:
    """
    A dataset class for testing images from common deep learning sources.
    
    This dataset provides a PyTorch-like interface for iterating through test images,
    with support for resetting and reusing the iterator.
    
    Args:
        image_sources (list, optional): List of (image_id, url) tuples. 
            If None, uses predefined IMAGENET_IMAGES.
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
        
        self.image_sources = image_sources if image_sources is not None else IMAGENET_IMAGES
        valid_ext = [".jpg", ".jpeg", '.JPEG', '.JPG']

        self.image_sources = [(img_id, url) for img_id, url, _ in self.image_sources if url.endswith(tuple(valid_ext))]
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

def to_shape(img_bytes, h = 224, w=224):
    assert isinstance(img_bytes, bytes), "Input must be raw image bytes"
    image = Image.open(BytesIO(img_bytes)).convert('RGB')
    
    # Resize image to 224x224
    image = image.resize((h, w))
    
    # Encode image back to bytes
    buffer = BytesIO()
    image.save(buffer, format='JPEG')
    return buffer.getvalue()

def get_timm_and_export_onnx(model_name, onnx_path = None, h = 224, w = 224):
    model, preprocessor = get_classification_model(model_name, h, w)
    export_x3hw(model, onnx_path, h, w)
    return model, preprocessor
    
class ClassifyModelTester:
    def __init__(self, model_name, onnx_path = None, h = 224, w = 224):

        self.model_name = model_name
        self.model, self.preprocessor = get_classification_model(self.model_name, h, w)
        self.h = h
        self.w = w
        if onnx_path:
            export_x3hw(self.model, onnx_path, h, w)
    
    def __call__(self, img_path):
        # image = Image.open(img_path).convert('RGB')
        with open(img_path, 'rb') as f:
            data = f.read()
        preprocessed = self.preprocessor(data).unsqueeze(0).to(next(self.model.parameters()).device)
        with torch.no_grad():
            result = torch.nn.functional.softmax(self.model(preprocessed).cuda(), dim=-1)
            index = torch.argmax(result).item()
            return index, result[0, index].item()
        
    def test(self, callable_func, fix_shape = False):
        all_result = {}
        dataset = TestImageDataset()
        for data_id, data in dataset:
            if data is not None:
                if fix_shape:
                    data =  to_shape(data,self.h, self.h)
                # preprocessed = self.preprocessor(data).unsqueeze(0)
                preprocessed = preprocess_with_cv2(data).unsqueeze(0)
                # import pdb;pdb.set_trace()
                
                with torch.no_grad():
                    result = torch.nn.functional.softmax(self.model(preprocessed).cuda(), dim=-1)
                    if callable_func:
                        extra_result = torch.nn.functional.softmax(callable_func(data), dim=-1)
                        all_result[data_id] = (result, extra_result)
                    else:
                        all_result[data_id] = (result, None)

                # import hami, numpy
                # data, req_size = hami.default_queue().get()
                # data = data['data']
                # data = data / 255.0
                # mean = torch.tensor([0.485, 0.456, 0.406], dtype=data.dtype, device=data.device)
                # std = torch.tensor([0.229, 0.224, 0.225], dtype=data.dtype, device=data.device)
                
                # if len(data.shape) == 3:
                #     data = data.permute(2, 0, 1).unsqueeze(0)
                # data = (data - mean[None, :, None, None]) / std[None, :, None, None]
                # a=torch.allclose(data.cuda(), preprocessed.cuda(), rtol=1e-2, atol=1e-2)
                # import pdb; pdb.set_trace()
            
        if callable_func:
            report_classification(all_result)
        return all_result
        
        
   
def test_from_raw_file(
    forward_function:  Callable[[List[tuple[str, bytes]]], Any],
    file_dir: str,
    num_clients=10,
    request_batch=1,
    total_number=10000,
    num_preload=1000,
    recursive=True,
    ext=[".jpg", ".JPG", ".jpeg", ".JPEG"],
):
    pass


if __name__ == "__main__":
    test_onnx()
    