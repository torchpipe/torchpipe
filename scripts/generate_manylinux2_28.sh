
set -e
rm -rf wheelhouse/*manylinux_2_28_x86_64.whl

rm -rf hami.egg-info/*

export PYVER=313

for PYVER in 38 39 310 311 312 313; do
    rm -rf .setuptools-cmake-build/
    rm -rf build/
    rm -rf dist/*
    # export PYVER=312
    export PATH="/opt/python/cp${PYVER}-cp${PYVER}/bin:$PATH"

    pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

    /opt/python/cp${PYVER}-cp${PYVER}/bin/python3  -m pip install build pybind11 auditwheel-symbols setuptools
    # python3 -m build
    USE_CXX11_ABI=1 /opt/python/cp${PYVER}-cp${PYVER}/bin/python3 setup.py -q bdist_wheel 
    # cibuildwheel --platform linux
    auditwheel repair --plat manylinux_2_28_x86_64 dist/hami-1.0.0a0-cp$PYVER-cp$PYVER-linux_x86_64.whl 
done

ldd .setuptools-cmake-build/hami/*.so