
set -e


# docker run --rm  --name hami2014 --gpus=all --ipc=host --network=host -v $(pwd):/workspace --shm-size 1G --ulimit memlock=-1 --ulimit stack=67108864  --privileged=true  -w /workspace \
#  -it hami:2014  bash
 


rm -rf wheelhouse/*manylinux_2_28_x86_64.whl

rm -rf hami*.egg-info/*

git config --global --add safe.directory .
git fetch --unshallow --tags 2>/dev/null || true

export PYVER=313

for PYVER in 38 39 310 311 312 313; do
    rm -rf .setuptools-cmake-build/
    rm -rf build/
    rm -rf dist/*.whl
    # export PYVER=312
    export PATH="/opt/python/cp${PYVER}-cp${PYVER}/bin:$PATH"

    pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

    /opt/python/cp${PYVER}-cp${PYVER}/bin/python3  -m pip install build pybind11 auditwheel-symbols setuptools setuptools_scm
    # python3 -m build
    USE_CXX11_ABI=1 /opt/python/cp${PYVER}-cp${PYVER}/bin/python3 setup.py -q bdist_wheel 
    auditwheel repair --plat manylinux_2_28_x86_64 dist/hami*-cp$PYVER-cp$PYVER-linux_x86_64.whl 
    # cibuildwheel --platform linux
    # for whl in dist/hami*-cp${PYVER}*-linux_x86_64.whl; do
    #     auditwheel repair --plat manylinux_2_28_x86_64 "$whl"
    # done

done

ldd .setuptools-cmake-build/hami/*.so