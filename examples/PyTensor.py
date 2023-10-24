# Copyright 2021-2023 NetEase.
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

import torchvision.transforms as T
transform = T.ToPILImage()
from torchvision.transforms import functional 
from PIL import Image
## pip install --upgrade pillow

## 或者， 更推荐（更快）：
## pip uninstall pillow && CC="cc -mavx2" pip install -U --force-reinstall pillow-simd
def tensor2tensor(data):
    assert (data.is_cpu)
    img = transform(data)
    print(img.size)
    img=img.resize((2,4))
    out= functional.pil_to_tensor(img)
    assert (out.is_cpu)
    
    return out