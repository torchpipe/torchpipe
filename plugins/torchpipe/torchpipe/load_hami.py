import os
import sys
import ctypes
from pathlib import Path
import importlib.util

def _find_hami_library():
    """Find and load hami library"""
    # 尝试直接导入 hami
    try:
        import hami._C
        return str(Path(hami._C.__file__).parent)
    except ImportError:
        # 如果 hami 没有安装，尝试查找相对路径
        candidates = [
            # 相对于当前包的路径
            Path(__file__).parent / "hami",
            # 相对于工作目录的路径
            Path.cwd() / ".setuptools-cmake-build" / "hami",
            # 其他可能的路径...
        ]
        
        # 获取当前 Python 版本的后缀
        ext_suffix = importlib.machinery.EXTENSION_SUFFIXES[0]  # 例如 '.cpython-310-x86_64-linux-gnu.so'
        lib_name = f"_C{ext_suffix}"
        
        for path in candidates:
            lib_path = path / lib_name
            if lib_path.exists():
                return str(path)
                
        raise ImportError(f"Cannot find hami library ({lib_name})")

def _find_hami_so():
    """Get the path to the hami library"""
    hami_dir = _find_hami_library()
    ext_suffix = importlib.machinery.EXTENSION_SUFFIXES[0]
    return (os.path.join(hami_dir, f'_C{ext_suffix}'))
            
def _load_hami():
    """Load hami library and add its directory to PATH"""
    hami_dir = _find_hami_library()
    print(hami_dir)
    
    # 添加到 PATH
    if sys.platform == 'linux':
        # Linux: 添加到 LD_LIBRARY_PATH
        old_path = os.environ.get('LD_LIBRARY_PATH', '')
        os.environ['LD_LIBRARY_PATH'] = f"{hami_dir}:{old_path}"
    elif sys.platform == 'darwin':
        # macOS: 添加到 DYLD_LIBRARY_PATH
        old_path = os.environ.get('DYLD_LIBRARY_PATH', '')
        os.environ['DYLD_LIBRARY_PATH'] = f"{hami_dir}:{old_path}"
    elif sys.platform == 'win32':
        # Windows: 添加到 PATH
        old_path = os.environ.get('PATH', '')
        os.environ['PATH'] = f"{hami_dir};{old_path}"

    # 预加载 hami 库
    try:
        if sys.platform == 'win32':
            ctypes.CDLL(os.path.join(hami_dir, '_C.pyd'))
        else:
            ext_suffix = importlib.machinery.EXTENSION_SUFFIXES[0]
            ctypes.CDLL(os.path.join(hami_dir, f'_C{ext_suffix}'))
    except Exception as e:
        raise ImportError(f"Failed to load hami library: {e}")

# 在导入时自动加载 hami 库
# _load_hami()

# 导入其他必要的模块和符号
# from . import _C
# ... 其他导入和初始化代码 ...