import os


def is_licensed(file):
    with open(file, "r") as f:
        lines = f.read()
        # print(len(lines))

        # if "any.hpp" in file:
        #     print(lines)
        if "Copyright" in lines or "copyright" in lines:
            ## ../torchpipe/csrc/backend/src_cuda/cudaUtility.h
            ## ../torchpipe/csrc/core/include/any.hpp
            if not "Torchpipe Authors" in lines and "NetEase" in lines:
                print(f"licensed without Torchpipe Authors: {file}")
                exit(0)
            return True

    return False


def write_cpp_license_header(filename):
    print(f"write {filename}")
    comment = """// Copyright 2021-2023 NetEase.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

"""
    # 读取原文件内容
    with open(filename, "r") as f:
        content = f.readlines()

    # 在开头添加注释
    content.insert(0, comment)

    # 将修改后的内容写回文件
    with open(filename, "w") as f:
        f.writelines(content)


def write_py_license_header(filename):
    print(f"write {filename}")
    comment = """# Copyright 2021-2023 NetEase.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
    # 读取原文件内容
    with open(filename, "r") as f:
        content = f.readlines()

    # 在开头添加注释
    content.insert(0, comment)

    # 将修改后的内容写回文件
    with open(filename, "w") as f:
        f.writelines(content)


def update_cpp_license(dir_path, check=False):
    assert os.path.exists(dir_path)
    all_files = []
    for root, dirs, files in os.walk(dir_path):
        path_for_cpp = [
            os.path.join(root, file) for file in files if file.endswith(".cpp")
        ]
        path_for_hpp = [
            os.path.join(root, file) for file in files if file.endswith(".hpp")
        ]
        path_for_h = [os.path.join(root, file) for file in files if file.endswith(".h")]

        all_files += path_for_cpp
        all_files += path_for_hpp
        all_files += path_for_h

    for file in all_files:
        if "thirdpart" in file:
            continue
        if is_licensed(file):
            # print(f"escape {file}")
            continue
        elif check:
            print(file)
        else:
            write_cpp_license_header(file)


def update_py_license(dir_path, check=False):
    assert os.path.exists(dir_path)
    all_files = []
    for root, dirs, files in os.walk(dir_path):
        path_for_py = [
            os.path.join(root, file) for file in files if file.endswith(".py")
        ]

        all_files += path_for_py

    for file in all_files:
        if "thirdpart" in file:
            continue
        if is_licensed(file):
            # print(f"escape {file}")
            continue
        elif check:
            print(file)
        else:
            write_py_license_header(file)


def update_all(check_only=True):
    target_dirs = ["../torchpipe", "../examples", "../test"]
    for target in target_dirs:
        update_cpp_license(target, check_only)
        update_py_license(target, check_only)


if __name__ == "__main__":
    update_all(check_only=False)
    # dir = "../torchpipe"
    # update_cpp_license(dir)

    # dir = "../torchpipe"
    # update_py_license(dir)

    # dir = "../examples"
    # update_cpp_license(dir)

    # dir = "../examples"
    # update_py_license(dir)

    # dir = "../test"
    # update_cpp_license(dir)

    # dir = "../test"
    # update_py_license(dir)
