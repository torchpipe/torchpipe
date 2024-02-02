mkdir -p /opt/intel
mkdir -p ~/.cache

cur=$(pwd)

cd ~/.cache

if [ ! -d "/opt/intel/openvino_2023.3.0" ]; then
    curl -L https://storage.openvinotoolkit.org/repositories/openvino/packages/2023.3/linux/l_openvino_toolkit_ubuntu20_2023.3.0.13775.ceeafaf64f3_x86_64.tgz  --output openvino_2023.3.0.tgz
    tar -xf openvino_2023.3.0.tgz
    mv  l_openvino_toolkit_ubuntu20_2023.3.0.13775.ceeafaf64f3_x86_64 /opt/intel/openvino_2023.3.0
fi



if [ ! -d "/opt/intel/openvino_2023" ]; then
    ln -s /opt/intel/openvino_2023.3.0 /opt/intel/openvino_2023
fi

python3 -m pip install -r /opt/intel/openvino_2023/python/requirements.txt

if ! dpkg -s libtbb2 >/dev/null 2>&1; then
    apt-get update -y && apt-get install libtbb2 -y
fi

source /opt/intel/openvino_2023/setupvars.sh

rm -r ~/.cache/*

cd ${cur}

echo $(python -c "import openvino; print('openvino version: ', openvino.__version__)")

# WITH_OPENCV=1 WITH_OPENVINO=1 DEBUG=1 pip install -e .