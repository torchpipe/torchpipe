from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDA_HOME
import torch
import glob
import os
import sys
from tqdm import tqdm
import shutil

TrtAddr = {
    "cuda11.8": "https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/10.8.0/tars/TensorRT-10.8.0.43.Linux.x86_64-gnu.cuda-11.8.tar.gz",
    "cuda12.8": "https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/10.8.0/tars/TensorRT-10.8.0.43.Linux.x86_64-gnu.cuda-12.8.tar.gz"
}

def get_trt_path():
    """Get TensorRT download URL based on CUDA version."""
    if not torch.cuda.is_available():
        raise Exception("CUDA is not available. TensorRT requires CUDA.")
    
    cuda_version = torch.version.cuda
    if cuda_version.startswith("11"):
        trt_path = TrtAddr["cuda11.8"]
    elif cuda_version.startswith("12"):
        trt_path = TrtAddr["cuda12.8"]
    else:
        raise Exception("TensorRT is not supported for CUDA <= 10")
    return trt_path

def download_with_progress(url, local_path):
    """Download file with progress bar."""
    import requests
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(local_path, "wb") as f, tqdm(
        desc="Downloading TensorRT",
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for data in response.iter_content(chunk_size=1024):
            size = f.write(data)
            pbar.update(size)

def download_and_install_trt(
    install_dir: str = "/usr/local/",
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
    import tempfile
    tmp_dir = tempfile.gettempdir()
    os.makedirs(tmp_dir, exist_ok=True)
    print(f"Building in temporary directory: {tmp_dir}")
    
    # Get TensorRT download URL
    if cuda_version:
        if cuda_version.startswith("11"):
            trt_path = TrtAddr["cuda11.8"]
        elif cuda_version.startswith("12"):
            trt_path = TrtAddr["cuda12.8"]
        else:
            raise ValueError("cuda_version must be 11 or 12")
    else:
        trt_path = get_trt_path()
    
    print(f"Selected TensorRT download URL: {trt_path}")
    
    # Download TensorRT
    local_path = os.path.join(tmp_dir, os.path.basename(trt_path))
    print(f"Downloading TensorRT to {local_path}...")
    download_with_progress(trt_path, local_path)
    
    # Extract TensorRT
    print("Extracting TensorRT...")
    trt_extract_dir = os.path.join(tmp_dir, "TensorRT")
    if os.path.exists(trt_extract_dir):
        shutil.rmtree(trt_extract_dir)
    os.makedirs(trt_extract_dir)
    
    import tarfile
    with tarfile.open(local_path) as tar:
        tar.extractall(path=trt_extract_dir)
    
    # Get the extracted directory name
    extracted_dir = next(os.walk(trt_extract_dir))[1][0]
    trt_source_dir = os.path.join(trt_extract_dir, extracted_dir)
    
    # Install TensorRT
    print(f"Installing TensorRT to {install_dir}...")
    
    # Copy include files
    include_dst = os.path.join(install_dir, "include")
    os.makedirs(include_dst, exist_ok=True)
    shutil.copytree(os.path.join(trt_source_dir, "include"), include_dst, dirs_exist_ok=True)
    
    # Copy lib files
    lib_dst = os.path.join(install_dir, "lib")
    os.makedirs(lib_dst, exist_ok=True)
    shutil.copytree(os.path.join(trt_source_dir, "lib"), lib_dst, dirs_exist_ok=True)
    
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