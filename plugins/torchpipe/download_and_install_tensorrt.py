# from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDA_HOME
import torch
import glob
import os
import sys
from tqdm import tqdm
from typing import Optional
import shutil
from pathlib import Path
import logging

TrtAddr = {
    "cuda11.8": "https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/10.8.0/tars/TensorRT-10.8.0.43.Linux.x86_64-gnu.cuda-11.8.tar.gz",
    "cuda12.8": "https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/10.8.0/tars/TensorRT-10.8.0.43.Linux.x86_64-gnu.cuda-12.8.tar.gz",
    "cuda11.8/trt109": "https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/10.9.0/tars/TensorRT-10.9.0.34.Linux.x86_64-gnu.cuda-11.8.tar.gz",
    "cuda12.8/trt109": "https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/10.9.0/tars/TensorRT-10.9.0.34.Linux.x86_64-gnu.cuda-12.8.tar.gz",
    "cuda12/trt1014": "https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/10.14.1/tars/TensorRT-10.14.1.48.Linux.x86_64-gnu.cuda-12.9.tar.gz",
}

POSSIBLE_TENSORRT_LIB_DIR = set({"/usr/lib/x86_64-linux-gnu/", '/usr/lib'})
POSSIBLE_TENSORRT_INCLUDE_DIR = set({"/usr/include/", '/usr/include/x86_64-linux-gnu/'})
def get_trt_path():
    """Get TensorRT download URL based on CUDA version."""
    if not torch.cuda.is_available():
        raise Exception("CUDA is not available. TensorRT requires CUDA.")
    
    cuda_version = torch.version.cuda.split('.')[0]  # 获取主版本号
    if cuda_version == "11":
        trt_path = TrtAddr["cuda11.8/trt109"]
    elif cuda_version == "12":
        trt_path = TrtAddr["cuda12.8/trt109"]
    else:
        raise NotImplementedError(f"TensorRT not supported for CUDA version {cuda_version}")
    return trt_path

def download_with_progress(url, local_path):
    """Download file with progress bar, skip if local file exists and size matches."""
    import requests
    # 检查本地文件是否存在
    if os.path.exists(local_path):
        local_size = os.path.getsize(local_path)
        # 获取远程文件大小（避免下载完整内容）
        response = requests.head(url, allow_redirects=True)
        remote_size = int(response.headers.get('content-length', 0))
        
        # 如果大小一致，直接跳过
        if local_size == remote_size and remote_size != 0:
            print(f"File already exists and size matches, skipping download: {local_path}")
            return
    
    # 如果文件不存在或大小不匹配，执行下载
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(local_path, "wb") as f, tqdm(
        desc=f"Downloading {os.path.basename(local_path)}",
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for data in response.iter_content(chunk_size=1024):
            size = f.write(data)
            pbar.update(size)


def exist_return(install_dir):
    TENSORRT_INCLUDE = os.path.join(install_dir, "include/")
    TENSORRT_LIB = os.path.join(install_dir, "lib")
    trt_found =  (Path(TENSORRT_INCLUDE) / "NvInfer.h").exists() and ( Path(TENSORRT_LIB) / "libnvonnxparser.so").exists()

    if trt_found:
        logging.info(f" Tensorrt founded in {install_dir}.  Setting it through TENSORRT_INCLUDE and TENSORRT_LIB")
        return TENSORRT_INCLUDE, TENSORRT_LIB
    else:
        return None, None
        
def download_and_install_trt(
    install_dir: Optional[str] = None,
    cleanup: bool = True,
    cuda_version: str = None
):
    """
    Download and install TensorRT.
    
    Args:
        install_dir: Installation directory (default: /usr/local/)
        cleanup: Whether to remove downloaded files after installation (default: True)
        cuda_version: Force specific CUDA version (11 or 12, default: auto-detect)
    """
    # Create temporary directory

    if install_dir is None:
        install_dir = "/usr/local/"
        TENSORRT_INCLUDE, TENSORRT_LIB = exist_return(install_dir)
        if TENSORRT_INCLUDE and TENSORRT_LIB:
            return TENSORRT_INCLUDE, TENSORRT_LIB
    
        
    for dir_path in POSSIBLE_TENSORRT_INCLUDE_DIR:
        if os.path.exists(os.path.join(dir_path, "NvInfer.h")):
            TENSORRT_INCLUDE = dir_path
            break
    for dir_path in POSSIBLE_TENSORRT_LIB_DIR:
        if os.path.exists(os.path.join(dir_path, "libnvonnxparser.so")):
            TENSORRT_LIB = dir_path
            break
    # print((os.path.join(dir_path, "NvInfer.h")), os.path.exists(os.path.join(dir_path, "libnvonnxparser.so")))
    if TENSORRT_INCLUDE and TENSORRT_LIB:
        return TENSORRT_INCLUDE, TENSORRT_LIB

    if not os.path.exists(install_dir):
        try:
            os.makedirs(install_dir, exist_ok=True)
        except:
            pass
    if not os.access(install_dir, os.W_OK):
        new_install_dir = os.path.expanduser("~/tensorrt_install")
        print(f"No write permission for {install_dir}. Using {new_install_dir} instead.")
        install_dir = new_install_dir

        TENSORRT_INCLUDE, TENSORRT_LIB = exist_return(install_dir)
        if TENSORRT_INCLUDE and TENSORRT_LIB:
            return TENSORRT_INCLUDE, TENSORRT_LIB

    import tempfile
    tmp_dir = tempfile.gettempdir()
    os.makedirs(tmp_dir, exist_ok=True)
    print(f"Building in temporary directory: {tmp_dir}")
    
    # Get TensorRT download URL
    if cuda_version:
        if cuda_version.startswith("11"):
            trt_path = TrtAddr["cuda11.8/trt109"]
        elif cuda_version.startswith("12"):
            trt_path = TrtAddr["cuda12.8/trt109"]
        else:
            raise ValueError("cuda_version must be 11 or 12")
    else:
        trt_path = get_trt_path()
    
    print(f"Selected TensorRT download URL: {trt_path}")
    
    # Download TensorRT
    local_path = os.path.join(tmp_dir, os.path.basename(trt_path))
    print(f"Find No TENSORRT_INCLUDE/TENSORRT_LIB. Downloading TensorRT to {local_path}...")
    download_with_progress(trt_path, local_path)
    
    # Extract TensorRT
    
    trt_extract_dir = os.path.join(tmp_dir, "TensorRT")
    print(f"Extracting TensorRT to {trt_extract_dir} ...")

    os.makedirs(trt_extract_dir, exist_ok=True)
    
    import tarfile
    with tarfile.open(local_path) as tar:
        tar.extractall(path=trt_extract_dir)
    
    # Get the extracted directory name
    extracted_dir = next(os.walk(trt_extract_dir))[1][0]
    trt_source_dir = os.path.join(trt_extract_dir, extracted_dir)
    
    # Install TensorRT
    print(f"Installing TensorRT to {install_dir} ...")
    
    # Copy include files
    include_dst = os.path.join(install_dir, "include")
    os.makedirs(include_dst, exist_ok=True)
    shutil.copytree(os.path.join(trt_source_dir, "include"), include_dst, dirs_exist_ok=True)
    
    # Copy lib files
    lib_dst = os.path.join(install_dir, "lib")

    os.makedirs(lib_dst, exist_ok=True)
    shutil.copytree(os.path.join(trt_source_dir, "lib"), lib_dst, dirs_exist_ok=True)
    assert os.path.exists(os.path.join(lib_dst, 'libnvinfer_plugin.so'))
    
    # Set environment variables
    print("\nTensorRT installation complete!")
    print("\nPlease add the following to your environment:")
    print(f"export LD_LIBRARY_PATH={lib_dst}:$LD_LIBRARY_PATH")
    print(f"export LIBRARY_PATH={lib_dst}:$LIBRARY_PATH")
    print(f"export CPATH={include_dst}:$CPATH")
    
    # Cleanup
    if cleanup:
        print("\nCleaning up temporary files...")
        os.remove(local_path)
        shutil.rmtree(trt_extract_dir)
    
    return include_dst, lib_dst

if __name__ == "__main__":
    import fire
    fire.Fire(download_and_install_trt)