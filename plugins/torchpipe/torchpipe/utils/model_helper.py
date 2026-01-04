from __future__ import annotations

# Model_helper.py
from collections import defaultdict
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
    pass
    # print("OpenCV not found. Please install it to use this feature.")

import torch
import torch.nn as nn
try:
    from torchvision import transforms
except Exception as e:
    pass
    # print(
    #     f"Torchvision not found. Please install it to use this feature. e = {e}")
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


def import_or_install_package(package_name: str, import_name=None) -> None:
    """Dynamically import or install missing packages.

    Args:
        package_name: Name of the package to install/import
    """
    if import_name is None:
        import_name = package_name
    try:
        module = importlib.import_module(import_name)
    except ImportError:
        print(f"Installing required package: {package_name}")
        subprocess.run(["pip", "install", package_name], check=True)
        module = importlib.import_module(import_name)
    return module


def get_max_supported_onnx_opset():
    """获取当前 PyTorch 版本支持的 ONNX 最大 opset 版本"""
    # PyTorch 1.8+ 使用这个属性
    if hasattr(torch.onnx, 'ONNX_MAX_OPSET_VERSION'):
        return torch.onnx.ONNX_MAX_OPSET_VERSION

    # PyTorch 1.12+ 移除了上述属性，使用这个内部常量
    if hasattr(torch.onnx, '_constants'):
        from torch.onnx import _constants
        if hasattr(_constants, 'ONNX_TORCHSCRIPT_EXPORTER_MAX_OPSET'):
            return _constants.ONNX_TORCHSCRIPT_EXPORTER_MAX_OPSET
        elif hasattr(_constants, 'ONNX_MAX_OPSET'):
            return _constants.ONNX_MAX_OPSET
    
    torch_version = torch.__version__.split('+')[0]  # 移除可能的 CUDA 后缀
    major, minor = map(int, torch_version.split('.')[:2])

    if (major == 1 and minor >= 1.12) or major >= 2:
        return 17
    else:
        return 11

def export_n3hw(model: nn.Module, onnx_path: str, input_h: int, input_w: int) -> str:
    """Export model to optimized ONNX format with input shape [-1, 3, H, W].

    Args:
        model: PyTorch model to export
        input_h: Input height
        input_w: Input width

    Returns:
        Path to the exported ONNX model
    """
    # Ensure dependencies are installed
    for package in ["onnxslim", "onnx", 'onnxsim']:
        import_or_install_package(package)

    # Import required modules after installation
    import onnx

    # Generate dummy input
    original_device = next(model.parameters()).device
    original_dtype = next(model.parameters()).dtype
    dummy_input = torch.randn(1, 3, input_h, input_w, 
                             device=original_device,
                             dtype=original_dtype)
    
    tmp_onnx_path = tempfile.mktemp()
    

    # Export initial ONNX model
    torch.onnx.export(
        model,
        dummy_input,
        tmp_onnx_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        opset_version=get_max_supported_onnx_opset()
    )

    # Optimize with ONNX-Simplifier
    os.system(f"onnxsim {tmp_onnx_path} {onnx_path}")
    os.system(f'onnxslim {tmp_onnx_path} {tmp_onnx_path}')
    os.remove(tmp_onnx_path)

    print(f"Optimized ONNX model saved to: {onnx_path}")
    return onnx_path


onnx_export = export_n3hw


def preprocess_with_cv2(image_bytes, input_h=224, input_w=224):
    nparr = np.frombuffer(image_bytes, np.uint8)  # 转换为numpy数组
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # 转换为 OpenCV 的 BGR 格式

    image = cv2.resize(image, (input_w, input_h))

    image = image.transpose((2, 0, 1))  # HWC to CHW
    image = image.astype(np.float32) / 255.0  # 归一化到 [0, 1]

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
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
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])
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
    import_or_install_package('modelscope')
    import_or_install_package('datasets')
    import_or_install_package('addict')

    from modelscope.msdatasets import MsDataset
    from modelscope.utils.constant import DownloadMode

    # {'image:FILE': '*10.JPEG', 'category': 0}

    ms_val_dataset = MsDataset.load(
        'mini_imagenet100', namespace='tany0699',
        subset_name='default', split='validation',
        download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS)  # 加载验证集
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
    import_or_install_package('scikit-learn', 'sklearn')
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
        ('precision', lambda t, p: precision_score(
            t, p, average='macro', zero_division=0)),
        ('recall', lambda t, p: recall_score(
            t, p, average='macro', zero_division=0)),
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


import_or_install_package('tabulate')

def report_classification(all_results, top_k=5):
    """
    Analyze classification model performance with formatted reporting.

    Args:
        all_results (dict): Mapping of data_id to (label, prediction_tensor)
                            prediction_tensor shape: (1, 1000)
        top_k (int): Number of top predictions to consider for top-k accuracy

    Statistics reported:
        - Overall accuracy
        - Top-k accuracy
        - Per-class accuracy
        - Class distribution
        - Most confused class pairs
        - Samples with largest prediction confidence gaps
    """
    from tabulate import tabulate  # no-

    # Initialize metrics
    total = 0
    correct = 0
    top_k_correct = 0
    class_stats = defaultdict(lambda: {'count': 0, 'correct': 0})
    confusion_matrix = defaultdict(lambda: defaultdict(int))
    error_samples = []
    all_confidences = []

    for data_id, (label, pred_tensor) in all_results.items():
        # Ensure tensor is on CPU and flattened
        pred = pred_tensor.cpu().squeeze()
        label = int(label)

        # Get predicted class and top-k classes
        _, topk_preds = torch.topk(pred, k=top_k)
        pred_class = topk_preds[0].item()

        # Update counts
        total += 1
        class_stats[label]['count'] += 1
        correct += int(pred_class == label)
        top_k_correct += int(label in topk_preds)

        # Track confidence for correct predictions
        if pred_class == label:
            class_stats[label]['correct'] += 1
            all_confidences.append(pred[label].item())

        # Record errors
        if pred_class != label:
            error_samples.append({
                'data_id': data_id,
                'true_class': label,
                'pred_class': pred_class,
                'confidence': pred[pred_class].item(),
                'true_confidence': pred[label].item()
            })
            confusion_matrix[label][pred_class] += 1

    # Calculate basic metrics
    accuracy = correct / total if total > 0 else 0
    top_k_accuracy = top_k_correct / total if total > 0 else 0

    # Calculate per-class accuracy
    class_accuracies = []
    for cls in sorted(class_stats.keys()):
        stats = class_stats[cls]
        acc = stats['correct'] / stats['count'] if stats['count'] > 0 else 0
        class_accuracies.append((cls, stats['count'], acc))

    # Find most confused class pairs
    confused_pairs = []
    for true_cls in confusion_matrix:
        for pred_cls, count in confusion_matrix[true_cls].items():
            confused_pairs.append((true_cls, pred_cls, count))
    confused_pairs.sort(key=lambda x: x[2], reverse=True)

    # Generate report tables
    print("\n===== Classification Report =====")

    # Main statistics table
    main_table = [
        ["Total Samples", total],
        ["Accuracy", f"{accuracy:.2%}"],
        [f"Top-{top_k} Accuracy", f"{top_k_accuracy:.2%}"],
        ["Mean Confidence (Correct)",
         f"{np.mean(all_confidences):.2%}" if all_confidences else "N/A"],
    ]
    print(tabulate(main_table, headers=["Metric", "Value"], tablefmt="grid"))

    # Class accuracy table (top 10)
    class_table = []
    for cls, count, acc in sorted(class_accuracies, key=lambda x: x[2])[:10]:
        class_table.append([f"Class {cls}", count, f"{acc:.2%}"])
    print("\n===== Worst Performing Classes =====")
    print(tabulate(class_table, headers=[
          "Class", "Samples", "Accuracy"], tablefmt="grid"))

    # Confusion matrix summary
    if confused_pairs:
        confusion_table = []
        for true_cls, pred_cls, count in confused_pairs[:5]:
            confusion_table.append([
                f"{true_cls} → {pred_cls}",
                count,
                f"{count/total:.1%} of total",
                f"{count/class_stats[true_cls]['count']:.1%} of class"
            ])
        print("\n===== Most Common Misclassifications =====")
        print(tabulate(confusion_table,
                       headers=["Confusion", "Count", "Total%", "Class%"],
                       tablefmt="grid"))

    # Error samples table
    if error_samples:
        error_samples.sort(
            key=lambda x: x['confidence'] - x['true_confidence'], reverse=True)
        error_table = []
        for err in error_samples[:3]:
            error_table.append([
                err['data_id'],
                f"{err['true_class']} → {err['pred_class']}",
                f"{err['true_confidence']:.1%}",
                f"{err['confidence']:.1%}",
                f"{err['confidence'] - err['true_confidence']:.1%}"
            ])
        print("\n===== High Confidence Errors =====")
        print(tabulate(error_table,
                       headers=["ID", "Misclassification", "True Conf",
                                "Pred Conf", "Δ Conf"],
                       tablefmt="grid"))

    # Return metrics dictionary
    return {
        'total_samples': total,
        'accuracy': accuracy,
        'top_k_accuracy': top_k_accuracy,
        'class_accuracies': dict((cls, acc) for cls, _, acc in class_accuracies),
        'confusion_matrix': dict(confusion_matrix),
        'error_samples': error_samples
    }


def build_label_mapping(all_results):
    """建立原始标签（0-99）到预测类别（0-999）的映射表

    Args:
        all_results: 字典格式 {id: (原始标签, 预测张量)}

    Returns:
        label_map: 字典 {原始标签: 预测类别}
        confidence_map: 字典 {原始标签: 中位数置信度}
    """
    # 收集每个原始标签的所有预测结果
    label_predictions = defaultdict(list)
    for _, (label, pred) in all_results.items():
        # 确保处理 (1, 1000) 形状张量
        if isinstance(pred, torch.Tensor):
            if pred.dim() == 2 and pred.shape[0] == 1:
                pred = torch.argmax(pred, dim=-1).item()

        # 记录预测类别和置信度
        label_predictions[label].append(pred)

    # 计算每个标签的中位数预测类别
    label_map = {}
    confidence_map = {}
    all_preds = set()
    for label, preds in label_predictions.items():
        # 合并所有样本的预测结果
        preds.sort()

        # 计算中位数预测类别
        median_class = int(preds[len(preds)//2])
        # 计算preds含有median_class的个数
        count = 0
        for i in range(len(preds)):
            if preds[i] == median_class:
                count += 1
        if count < len(preds) // 2:
            median_class = -1
        else:
            all_preds.add(median_class)
        # 记录映射关系
        label_map[label] = (median_class)
    print(f"len(label_map): {len(label_map)}, label: {len(label_predictions)}")
    assert (len(label_map) == len(label_predictions))
    return label_map


def compare_classification(all_result):
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
            pytorch_prob = result[0, pytorch_class].item(
            ) if result.dim() > 1 else result[pytorch_class].item()
            onnx_prob = onnx_result[0, onnx_class].item(
            ) if onnx_result.dim() > 1 else onnx_result[onnx_class].item()
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
        pytorch_top_prob = result[0, pytorch_class].item(
        ) if result.dim() > 1 else result[pytorch_class].item()
        onnx_top_prob_for_pytorch_class = onnx_result[0, pytorch_class].item(
        ) if onnx_result.dim() > 1 else onnx_result[pytorch_class].item()
        top_class_diff = abs(pytorch_top_prob -
                             onnx_top_prob_for_pytorch_class)

        if top_class_diff > max_top_class_diff:
            max_top_class_diff = top_class_diff
            max_top_class_diff_id = data_id

        # Print individual result details
        if current_max_diff > 1e-6:
            print(f"Data ID: {data_id}")
            print(
                f"PyTorch Result: {result.shape} max = {result.max().item()}")
            print(
                f"user's Result: {onnx_result.shape} max = {onnx_result.max().item()}")
            print(f"Max Difference: {current_max_diff:.6f}")
            print("Value similarity: " +
                  ("Similar" if is_similar else "Different"))
            print("Class prediction: " + ("Same" if is_same_class else "Different") +
                  f" ({pytorch_class} vs {onnx_class})")
            print("-" * 50)

    # Sort mismatches by the absolute difference (largest first)
    class_mismatches.sort(key=lambda x: x['abs_diff'], reverse=True)

    # Calculate and print overall statistics
    value_similarity_pct = (similar_count / total_count) * \
        100 if total_count > 0 else 0
    class_match_pct = (same_class_count / total_count) * \
        100 if total_count > 0 else 0

    print("\n===== Overall Statistics =====")
    print(f"Total samples: {total_count}")
    print(
        f"Samples with similar values: {similar_count}/{total_count} ({value_similarity_pct:.2f}%)")
    print(
        f"Samples with same class prediction: {same_class_count}/{total_count} ({class_match_pct:.2f}%)")

    # Report largest differences
    print("\n===== Largest Differences =====")
    print(
        f"Sample with largest tensor difference: Data ID: {max_diff_data_id}, Max diff: {max_abs_diff:.6f}")
    print(
        f"Sample with largest top class probability difference: Data ID: {max_top_class_diff_id}, Diff: {max_top_class_diff:.6f}")

    # Report class prediction mismatches (up to 5)
    if class_mismatches:
        print("\n===== Top Class Prediction Mismatches =====")
        for i, mismatch in enumerate(class_mismatches[:5]):
            print(f"Mismatch {i+1}: Data ID {mismatch['data_id']}")
            print(
                f"  PyTorch: Class {mismatch['pytorch_class']} with prob {mismatch['pytorch_prob']:.6f}")
            print(
                f"  ONNX: Class {mismatch['onnx_class']} with prob {mismatch['onnx_prob']:.6f}")
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

        self.image_sources = [(img_id, url) for img_id, url,
                              _ in self.image_sources if url.endswith(tuple(valid_ext))]
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
                        self.logger.debug(
                            f"Retry {attempt+1}/{self.max_retries} for {full_id}: {str(e)}")
                    else:
                        self.logger.warning(
                            f"Failed to download {full_id}: {str(e)}")
                        self.failed_image_ids.append(full_id)

            yield full_id, image_bytes

    def _log_failures(self):
        """Log information about failed downloads."""
        if self.failed_image_ids:
            self.logger.error(
                f"Failed to download {len(self.failed_image_ids)} images:")
            for img_id in self.failed_image_ids:
                self.logger.error(f"  - {img_id}")

    @property
    def failed_count(self):
        """Return the number of images that failed to download."""
        return len(self.failed_image_ids)


def test_onnx():
    model, preprocessor = get_classification_model("resnet50", 224, 224)

    onnx_path = f"{model.__class__.__name__}.onnx"
    export_n3hw(model, onnx_path, 224, 224)

    onnx_model = OnnxModel(onnx_path, preprocessor)

    all_result = {}
    dataset = TestImageDataset()
    for data_id, data in TestImageDataset():
        if data is not None:
            preprocessed = preprocessor(data).unsqueeze(0)
            with torch.no_grad():
                result = torch.nn.functional.softmax(
                    model(preprocessed), dim=-1)
            onnx_result = torch.nn.functional.softmax(
                torch.from_numpy(onnx_model(data)[0]), dim=-1)
            all_result[data_id] = (result, onnx_result)
    compare_classification(all_result)


def to_shape(img_bytes, h=224, w=224):
    assert isinstance(img_bytes, bytes), "Input must be raw image bytes"
    image = Image.open(BytesIO(img_bytes)).convert('RGB')

    # Resize image to 224x224
    image = image.resize((h, w))

    # Encode image back to bytes
    buffer = BytesIO()
    image.save(buffer, format='JPEG')
    return buffer.getvalue()


def get_timm_and_export_onnx(model_name, onnx_path=None, h=224, w=224):
    model, preprocessor = get_classification_model(model_name, h, w)
    export_n3hw(model, onnx_path, h, w)
    return model, preprocessor


def analyze_profile(profile_results):
    """
    Analyze the profile results from multiple clients to compute performance metrics.

    Args:
        profile_results (list of dict): List of profiling results, each containing
            'thread_id', 'start_time', 'end_time', and 'request_id'.

    Returns:
        dict: A dictionary containing performance metrics including total requests,
            number of clients, QPS, average latency, TP50, TP99, and per-client stats.
    """
    from tabulate import tabulate

    # Group results by thread_id
    clients = {}
    for result in profile_results:
        tid = result['thread_id']
        if tid not in clients:
            clients[tid] = []
        clients[tid].append(result)

    total_requests = len(profile_results)
    num_clients = len(clients)

    # Calculate total time span (ensure no division by zero)
    start_times = [r['start_time']
                   for r in profile_results] if profile_results else []
    end_times = [r['end_time']
                 for r in profile_results] if profile_results else []
    min_start = min(start_times) if start_times else 0
    max_end = max(end_times) if end_times else 0
    total_time = max_end - min_start if start_times and end_times else 0

    # Compute QPS (requests per second)
    qps = total_requests / total_time if total_time > 0 else 0

    # Calculate latencies and percentiles
    latencies = [r['end_time'] - r['start_time']
                 for r in profile_results] if profile_results else []
    sorted_latencies = sorted(latencies)
    avg_latency = sum(latencies) / len(latencies) if latencies else 0

    def compute_percentile(sorted_data, percentile):
        """
        Compute the percentile value from a sorted list of data using linear interpolation.
        """
        if not sorted_data:
            return 0
        index = (len(sorted_data) - 1) * percentile / 100
        integer_part = int(index)
        fractional_part = index - integer_part
        if fractional_part == 0:
            return sorted_data[integer_part]
        else:
            lower = sorted_data[integer_part]
            upper = sorted_data[integer_part + 1]
            return lower + (upper - lower) * fractional_part

    tp50 = compute_percentile(sorted_latencies, 50)
    tp99 = compute_percentile(sorted_latencies, 99)

    # Build result dictionary
    result = {
        'total_requests': total_requests,
        'number_clients': num_clients,
        'total_time_sec': total_time,
        'qps': qps,
        'average_latency_sec': avg_latency,
        'tp50_latency_sec': tp50,
        'tp99_latency_sec': tp99,
        'client_stats': {tid: len(res) for tid, res in clients.items()}
    }

    # Print per-client request counts
    client_table = []
    for tid, res in clients.items():
        client_table.append([f"Client {tid}", len(res)])
    print("\n")
    # print(tabulate(client_table, headers=["Client ID", "Requests"], tablefmt="grid"))

    # Print summary metrics in a formatted table
    summary_table = [
        ["Total Requests", result['total_requests']],
        ["Number of Clients", result['number_clients']],
        ["Total Time (s)", f"{result['total_time_sec']:.2f}"],
        ["QPS (requests/sec)", f"{result['qps']:.2f}"],
        ["Average Latency (s)", f"{result['average_latency_sec']:.4f}"],
        ["TP50 Latency (s)", f"{result['tp50_latency_sec']:.4f}"],
        ["TP99 Latency (s)", f"{result['tp99_latency_sec']:.4f}"]
    ]
    print(tabulate(summary_table, headers=[
          "Metric", "Value"], tablefmt="grid"))

    return result


class ClassifyModelTester:
    def __init__(self, model_name, onnx_path=None, h=224, w=224):

        self.model_name = model_name
        self.model, self.preprocessor = get_classification_model(
            self.model_name, h, w)
        self.h = h
        self.w = w
        if onnx_path:
            export_n3hw(self.model, onnx_path, h, w)

    def __call__(self, img_path):
        # image = Image.open(img_path).convert('RGB')
        with open(img_path, 'rb') as f:
            data = f.read()
        preprocessed = self.preprocessor(data).unsqueeze(
            0).to(next(self.model.parameters()).device)
        with torch.no_grad():
            result = torch.nn.functional.softmax(
                self.model(preprocessed).cuda(), dim=-1)
            index = torch.argmax(result).item()
            return index, result[0, index].item()

    def test(self, callable_func, fix_shape=False):
        all_result = {}
        dataset = TestImageDataset()
        for data_id, data in dataset:
            if data is not None:
                if fix_shape:
                    data = to_shape(data, self.h, self.h)
                # preprocessed = self.preprocessor(data).unsqueeze(0)
                preprocessed = preprocess_with_cv2(data).unsqueeze(0)
                # import pdb;pdb.set_trace()

                with torch.no_grad():
                    result = torch.nn.functional.softmax(
                        self.model(preprocessed).cuda(), dim=-1)
                    if callable_func:
                        extra_result = torch.nn.functional.softmax(
                            callable_func(data), dim=-1)
                        all_result[data_id] = (result, extra_result)
                    else:
                        all_result[data_id] = (result, None)

                # import omniback, numpy
                # data, req_size = omniback.default_queue().get()
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
            compare_classification(all_result)
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
