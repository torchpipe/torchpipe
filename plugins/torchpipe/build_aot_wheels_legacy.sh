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

rm -f "$torchpipe"/torchpipe/lib/*.so

ls -la "$omniback"/.git
git config --global --add safe.directory "$omniback"
# git describe --tags

mkdir -p "$omniback"/.venv

uv venv "$omniback"/.venv/py3.9 --python 3.9
source "$omniback"/.venv/py3.9/bin/activate
uv pip install build

cd "$torchpipe"
mkdir -p wheelhouse/
rm -rf dist/
python -m build -w 
ls dist
if [[ "$os" == "Linux" ]]; then
    # python -m wheel tags dist/*.whl --python-tag="$python_version" --abi-tag="$python_version" --remove
    ls dist/*.whl
    cp dist/*.whl wheelhouse/

fi
ls wheelhouse