set -e

export PYVER=313

export PATH="/opt/python/cp${PYVER}-cp${PYVER}/bin:$PATH"
pip install wheelhouse/omniback*-cp${PYVER}-cp$PYVER-*manylinux_2_28_x86_64.whl

cd tests && pip install -r requirements.txt && pytest