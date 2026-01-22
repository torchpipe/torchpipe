#!/bin/bash
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# shellcheck disable=SC1090,1091
set -eux

arch=$1


os=$(uname -s)

case "$os" in
    "Linux" | "Darwin")
        ;;
    *)
        echo "Unknown OS: $os"
        return 1
        ;;
esac

export UV_VENV_CLEAR=1
export TVM_FFI_DISABLE_TORCH_C_DLPACK=1

omniback="$PWD"
torchpipe="$omniback"/plugins/torchpipe
csrc="$omniback"/plugins/torchpipe/torchpipe

rm -f "$torchpipe"/torchpipe/lib/*.so

#   sed -e 's|^mirrorlist=|#mirrorlist=|g' \
#       -e 's|^# baseurl=https://repo.almalinux.org|baseurl=https://mirrors.aliyun.com|g' \
#       -i.bak \
#       /etc/yum.repos.d/almalinux*.repo
    
# dnf makecache  

# # see https://developer.aliyun.com/mirror/almalinux?spm=a2c6h.13651102.0.0.62181b11JTaroF


ls -la "$omniback"/.git
git config --global --add safe.directory "$omniback"
# git describe --tags


function build_local_libs() {
    local torch_version=$1
    local python_version=$2

    
    source "$omniback"/.venv/py"$python_version"/bin/activate
    # if [[ "$os" == "Linux" ]]; then
    #     uv pip install torch=="$torch_version" --index-url "$(get_torch_url "$torch_version")"
    # else
    #     uv pip install torch=="$torch_version"
    # fi
    uv pip install torch==$torch_version # -i  http://mirrors.aliyun.com/pypi/simple/
    
    if [[ "$os" == "Linux" ]]; then
        python -m omniback.utils.build_lib --output-dir "$torchpipe"/torchpipe/lib --source-dirs "$csrc"/csrc/torchplugins/ "$csrc"/csrc/helper/ --include-dirs="$csrc"/csrc/ --build-with-cuda --name torchpipe_core
    fi
    ls "$torchpipe"/torchpipe/lib
    deactivate
    rm -rf "$omniback"/.venv/torch"$torch_version"

}

mkdir -p "$omniback"/.venv
mkdir -p "$torchpipe"/lib

uv venv "$omniback"/.venv/py3.9 --python 3.9
source "$omniback"/.venv/py3.9/bin/activate
uv pip install setuptools ninja fire
uv pip install -v .
deactivate

uv venv "$omniback"/.venv/py3.11 --python 3.11
source "$omniback"/.venv/py3.11/bin/activate
uv pip install setuptools ninja fire
rm -rf build
uv pip install -v .
deactivate

# https://pytorch.org/get-started/previous-versions/
torch_versions=("1.13" "2.0" "2.1")
for version in "${torch_versions[@]}"; do
    build_local_libs "$version" 3.9
done

"2.3" "2.4" "2.5" "2.6" "2.7" "2.8" "2.9" # => next version
for version in "${torch_versions[@]}"; do
    build_local_libs "$version" 3.11
done


# cp "$omniback"/lib/*.so "$torchpipe"/torchpipe
source "$omniback"/.venv/py3.9/bin/activate
uv pip install build wheel scikit_build_core setuptools-scm
cd "$torchpipe"
mkdir -p wheelhouse/
rm -rf dist/
python -m build -w --no-isolation
ls dist
if [[ "$os" == "Linux" ]]; then
    # python -m wheel tags dist/*.whl --python-tag="$python_version" --abi-tag="$python_version" --remove
    uv pip install auditwheel
    ls dist/*.whl
    # cp dist/*.whl wheelhouse/
    auditwheel repair --exclude libomniback.so --exclude libomniback_cxx03.so --exclude libtvm_ffi.so \
        --exclude libtorch.so --exclude libtorch_cpu.so --exclude libc10.so --exclude libtorch_python.so --exclude libtorch_cuda.so --exclude libc10_cuda.so dist/*.whl -w wheelhouse
else
    # python -m wheel tags dist/*.whl --python-tag="$python_version" --abi-tag="$python_version" --platform-tag=macosx_11_0_arm64 --remove
    uv pip install delocate
    delocate-wheel -v --ignore-missing-dependencies --exclude libtorch.dylib,libtorch_cpu.dylib,libc10.dylib,libtorch_python.dylib dist/*.whl -w wheelhouse
fi
ls wheelhouse