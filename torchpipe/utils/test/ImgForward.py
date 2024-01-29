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

# from .Forward import Forward
# import os
# import cv2
# from typing import List, Tuple
# import random
# import numpy as np
# # , ext=[".jpg", '.JPG', '.jpeg', '.JPEG']

# class JpgForward(Forward):
#     def __init__(self, data: List, request_batch=1):
#         super().__init__()
#         self.data = data
#         self.request_batch = request_batch

#     def __call__(self):
#         data = random.sample(self.data, self.request_batch)
#         self.forward(data)


    
#     def forward(self, data: List[Tuple[str, np.ndarray]]):
#         raise RuntimeError("Requires users to implement this function")

    
#     def preload(img_dir, target_number= 100, ext=[".jpg", '.JPG', '.jpeg', '.JPEG']) -> List[Tuple[str, np.ndarray]]:
#         assert(target_number > 0 and target_number < 10000)
#         img_root = img_dir
#         if not os.path.exists(img_root):
#             raise RuntimeError(img_dir+" not exists")


#         list_images = []
#         result = []
#         for root, folders, filenames in os.walk(img_root):
#             for filename in filenames:
#                 if os.path.splitext(filename)[-1] in ext:
#                     list_images.append(os.path.join(root, filename))
#         for img_path in  list_images:
#             with open(img_path, 'rb') as f:
#                 img=f.read()
#             if img: 
#                 result.append((img_path, img)) 
#                 if len(result) == target_number:
#                     break
#         if len(result) == 0:
#             raise RuntimeError("find no vaild imgs. ext = "+ext)
#         while len(result) < target_number:    
#             result.append(random.choice(result))
#         return result
