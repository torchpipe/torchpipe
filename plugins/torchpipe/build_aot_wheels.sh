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
python_version=3.8

os=$(uname -s)

case "$os" in
    "Linux" | "Darwin")
        ;;
    *)
        echo "Unknown OS: $os"
        return 1
        ;;
esac

omniback="$PWD"
torchpipe="$omniback"/plugins/torchpipe

ls -la "$omniback"/.git
git config --global --add safe.directory "$omniback"
git describe --tags

function get_torch_url() {
    local version="$1"
    case "$version" in
        "2.4" | "2.5")
            echo "https://download.pytorch.org/whl/cu124"
            ;;
        "2.6")
            echo "https://download.pytorch.org/whl/cu126"
            ;;
        "2.7")
            echo "https://download.pytorch.org/whl/cu128"
            ;;
        "2.8" | "2.9")
            echo "https://download.pytorch.org/whl/cu129"
            ;;
        *)
            echo "Unknown or unsupported torch version: $version" >&2
            return 1
            ;;
    esac
}


function build_local_libs() {
    local torch_version=$1

    uv venv "$omniback"/.venv/torch"$torch_version" --python "$python_version"
    source "$omniback"/.venv/torch"$torch_version"/bin/activate
    uv pip install setuptools ninja
    if [[ "$os" == "Linux" ]]; then
        uv pip install torch=="$torch_version" --index-url "$(get_torch_url "$torch_version")"
    else
        uv pip install torch=="$torch_version"
    fi
    uv pip install -v .
    python -m omniback.utils.build_libs --output-dir "$omniback"/lib
    if [[ "$os" == "Linux" ]]; then
        python -m omniback.utils.build_libs --output-dir "$omniback"/lib --build-with-cuda
    fi
    ls "$omniback"/lib
    deactivate
    rm -rf "$omniback"/.venv/torch"$torch_version"

}

mkdir -p "$omniback"/.venv
mkdir -p "$omniback"/lib

# torch_versions=("2.4" "2.5" "2.6" "2.7" "2.8" "2.9")
# for version in "${torch_versions[@]}"; do
#     build_local_libs "$version"
# done

# cp "$omniback"/lib/*.so "$torchpipe"/torchpipe
uv venv "$omniback"/.venv/build --python "$python_version"
source "$omniback"/.venv/build/bin/activate
uv pip install build wheel
cd "$torchpipe"
mkdir -p wheelhouse/
rm -rf dist/
python -m build -w
ls dist
if [[ "$os" == "Linux" ]]; then
    # python -m wheel tags dist/*.whl --python-tag="$python_version" --abi-tag="$python_version" --remove
    uv pip install auditwheel
    ls dist/*.whl
    cp dist/*.whl wheelhouse/
    # auditwheel repair --exclude libtorch.so --exclude libtorch_cpu.so --exclude libc10.so --exclude libtorch_python.so --exclude libtorch_cuda.so --exclude libc10_cuda.so dist/*.whl -w wheelhouse
else
    # python -m wheel tags dist/*.whl --python-tag="$python_version" --abi-tag="$python_version" --platform-tag=macosx_11_0_arm64 --remove
    uv pip install delocate
    delocate-wheel -v --ignore-missing-dependencies --exclude libtorch.dylib,libtorch_cpu.dylib,libc10.dylib,libtorch_python.dylib dist/*.whl -w wheelhouse
fi
ls wheelhouse