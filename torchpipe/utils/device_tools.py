# Copyright 2021-2024 NetEase.
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


def install_package(package, version="upgrade"):
    from sys import executable
    from subprocess import check_call

    result = False
    if version.lower() == "upgrade":
        result = check_call(
            [executable, "-m", "pip", "install", package, "--upgrade", "--user"]
        )
    else:
        from pkg_resources import get_distribution

        current_package_version = None
        try:
            current_package_version = get_distribution(package)
        except Exception:
            pass
        if current_package_version is None or current_package_version != version:
            installation_sign = "==" if ">=" not in version else ""
            result = check_call(
                [
                    executable,
                    "-m",
                    "pip",
                    "install",
                    package + installation_sign + version,
                    "--user",
                ]
            )
    return result


import GPUtil

gpus = GPUtil.getGPUs()


# NVIDIA GeForce GTX 1080 Ti


def name():
    return gpus[0].name


def driver():
    return gpus[0].driver
