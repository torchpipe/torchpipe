import os
import requests
import zipfile
import subprocess
from pathlib import Path

OPENCV_VERSION = os.environ.get("OPENCV_VERSION", "4.12.0")


def download(output_dir=None):
    """
    构建 OpenCV
    
    Args:
        output_dir (str | Path | None): 构建输出目录（CMake build directory）。
            若为 None，则使用源码目录下的 build 子目录。
    
    Returns:
        Path: OpenCV 构建完成后的安装根目录
    """
    # 修复：移除 URL 中多余空格
    OPENCV_URL = f"https://codeload.github.com/opencv/opencv/zip/refs/tags/{OPENCV_VERSION}"
    OPENCV_ZIP = f"opencv-{OPENCV_VERSION}.zip"

    cache_dir = Path(output_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    zip_path = cache_dir / OPENCV_ZIP
    zip_path_cache = cache_dir / (OPENCV_ZIP + ".cache")

    # 下载 OpenCV 源码（若不存在）
    if not zip_path.exists():
        print(f"Downloading {OPENCV_URL} to {zip_path}")
        print("You may manually download it if it is too slow.")
        response = requests.get(OPENCV_URL, stream=True)
        response.raise_for_status()
        with open(zip_path_cache, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        zip_path_cache.rename(zip_path)

    # 解压源码
    print(f"Extracting {OPENCV_ZIP} to {cache_dir}...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(cache_dir)



# 示例用法
if __name__ == "__main__":
    import fire
    fire.Fire(download)