
set -e
rm -rf wheelhouse/*manylinux2014_x86_64.whl

rm -rf hami*.egg-info/*

git config --global --add safe.directory .

export PYVER=313

for PYVER in 38 39 310 311 312 313; do
    rm -rf .setuptools-cmake-build/
    rm -rf build/
    rm -rf dist/*.whl
    # export PYVER=312
    export PATH="/opt/python/cp${PYVER}-cp${PYVER}/bin:$PATH"

    pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

    /opt/python/cp${PYVER}-cp${PYVER}/bin/python3  -m pip install build pybind11 auditwheel-symbols setuptools setuptools_scm
    USE_CXX11_ABI=0 /opt/python/cp${PYVER}-cp${PYVER}/bin/python3 setup.py -q bdist_wheel 

    auditwheel repair --plat manylinux2014_x86_64 dist/hami*-cp$PYVER-cp$PYVER-linux_x86_64.whl 

done

ldd .setuptools-cmake-build/hami/*.so