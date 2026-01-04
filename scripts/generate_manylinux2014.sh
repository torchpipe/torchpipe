
set -e
rm -rf wheelhouse/*manylinux2014_x86_64.whl

rm -rf omniback*.egg-info/*

git config --global --add safe.directory .

export PYVER=313

for PYVER in 38 39 310 311 312 313; do
    rm -rf .setuptools-cmake-build/
    rm -rf build/
    rm -rf dist/*.whl
    # export PYVER=312
    export PATH="/opt/python/cp${PYVER}-cp${PYVER}/bin:$PATH"

    pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

    /opt/python/cp${PYVER}-cp${PYVER}/bin/python3 -m pip install --upgrade \
    pip \
    build \
    wheel \
    scikit-build-core \
    ninja \
    pybind11[global] \
    setuptools-scm \
    auditwheel \
    auditwheel-symbols
    USE_CXX11_ABI=0 /opt/python/cp${PYVER}-cp${PYVER}/bin/python3 -m build --wheel --no-isolation
    auditwheel repair --plat manylinux2014_x86_64 dist/omniback*-cp$PYVER-cp$PYVER-linux_x86_64.whl 

done

ldd build/*.so