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



def platform_has_fast_fp16():
    import tensorrt as trt
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    a=trt.Builder(TRT_LOGGER)
    return a.platform_has_fast_fp16

if __name__ == "__main__":
    print(platform_has_fast_fp16())