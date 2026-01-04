import os
import sys
import ctypes
from pathlib import Path
import importlib.util

def _find_omniback_library():
    """Find and load omniback library"""
    # 尝试直接导入 omniback
    try:
        import omniback._C
        return str(Path(omniback._C.__file__).parent)
    except ImportError:
        # 如果 omniback 没有安装，尝试查找相对路径
        candidates = [
            # 相对于当前包的路径
            Path(__file__).parent / "omniback",
            # 相对于工作目录的路径
            Path.cwd() / ".setuptools-cmake-build" / "omniback",
            # 其他可能的路径...
        ]
        
        # 获取当前 Python 版本的后缀
        ext_suffix = importlib.machinery.EXTENSION_SUFFIXES[0]  # 例如 '.cpython-310-x86_64-linux-gnu.so'
        lib_name = f"_C{ext_suffix}"
        
        for path in candidates:
            lib_path = path / lib_name
            if lib_path.exists():
                return str(path)
                
        raise ImportError(f"Cannot find omniback library ({lib_name})")

def _find_omniback_so():
    """Get the path to the omniback library"""
    omniback_dir = _find_omniback_library()
    ext_suffix = importlib.machinery.EXTENSION_SUFFIXES[0]
    return (os.path.join(omniback_dir, f'_C{ext_suffix}'))
            
def _load_omniback():
    """Load omniback library and add its directory to PATH"""
    omniback_dir = _find_omniback_library()
    print(omniback_dir)
    
    # 添加到 PATH
    if sys.platform == 'linux':
        # Linux: 添加到 LD_LIBRARY_PATH
        old_path = os.environ.get('LD_LIBRARY_PATH', '')
        os.environ['LD_LIBRARY_PATH'] = f"{omniback_dir}:{old_path}"
    elif sys.platform == 'darwin':
        # macOS: 添加到 DYLD_LIBRARY_PATH
        old_path = os.environ.get('DYLD_LIBRARY_PATH', '')
        os.environ['DYLD_LIBRARY_PATH'] = f"{omniback_dir}:{old_path}"
    elif sys.platform == 'win32':
        # Windows: 添加到 PATH
        old_path = os.environ.get('PATH', '')
        os.environ['PATH'] = f"{omniback_dir};{old_path}"

    # 预加载 omniback 库
    try:
        if sys.platform == 'win32':
            ctypes.CDLL(os.path.join(omniback_dir, '_C.pyd'))
        else:
            ext_suffix = importlib.machinery.EXTENSION_SUFFIXES[0]
            ctypes.CDLL(os.path.join(omniback_dir, f'_C{ext_suffix}'))
    except Exception as e:
        raise ImportError(f"Failed to load omniback library: {e}")

# 在导入时自动加载 omniback 库
# _load_omniback()

# 导入其他必要的模块和符号
# from . import _C
# ... 其他导入和初始化代码 ...