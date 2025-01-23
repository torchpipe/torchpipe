
import math
import torch
import torchpipe as tp

from transformers.models.llava_next.modeling_llava_next import get_anyres_image_grid_shape
from transformers.models.llava_onevision.modeling_llava_onevision import unpad_image
class PackImageFeatures:
    def init(self, config) -> bool:
        print(f"Initalized PackImageFeatures with config:",  config)
        self.image_size = 384
        self.image_grid_pinpoints = [[384, 384], [384, 768], [384, 1152], [384, 1536], [384, 1920], [384, 2304], [768, 384], [768, 768], [768, 1152], [768, 1536], [768, 1920], [768, 2304], [1152, 384], [1152, 768], [1152, 1152], [1152, 1536], [1152, 1920], [1152, 2304], [1536, 384], [1536, 768], [1536, 1152], [1536, 1536], [1536, 1920], [1536, 2304], [1920, 384], [1920, 768], [1920, 1152], [1920, 1536], [1920, 1920], [1920, 2304], [2304, 384], [2304, 768], [2304, 1152], [2304, 1536], [2304, 1920], [2304, 2304]]
        self.vision_aspect_ratio = 9
        self.image_newline = torch.load('model_files/image_newline.pt', map_location=torch.device('cuda'))  
        self.patch_size = 14
        assert self.image_newline.dtype == torch.float16 
         
        self.image_token_index = 151646
        return True
    
    def forward(self, input: tp._C.Dict) -> None:
        # torch.Size([5, 729, 896])
        image_feature = input['data']
        img_h = input['img_h']
        img_w = input['img_w']
        # print(img_h, img_w)
        
        ori_image_size = (img_h, img_w)
        
        if image_feature.shape[0] > 1:
            base_image_feature = image_feature[0]
            image_feature = image_feature[1:]
            height = width = self.image_size // self.patch_size
            if height * width != base_image_feature.shape[0]:
                raise ValueError("The number of patches is not consistent with the image size.")
            num_patch_height, num_patch_width = get_anyres_image_grid_shape(
                ori_image_size,
                self.image_grid_pinpoints,
                self.image_size,
            )
            image_feature = image_feature.view(num_patch_height, num_patch_width, height, width, -1)
            image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
            image_feature = image_feature.flatten(1, 2).flatten(2, 3)
            image_feature = unpad_image(image_feature, ori_image_size)
            max_num_patches = self.vision_aspect_ratio
            channels, curr_height, curr_width = image_feature.shape
            ratio = math.sqrt(curr_height * curr_width / (max_num_patches * height**2))
            if ratio > 1.1:
                image_feature = image_feature[None]
                image_feature = torch.nn.functional.interpolate(
                    image_feature, [int(curr_height // ratio), int(curr_width // ratio)], mode="bilinear"
                )[0]
            if self.image_newline is not None:
               
                image_feature = torch.cat(
                    (
                        image_feature,
                        self.image_newline[:, None, None]
                        .expand(*image_feature.shape[:-1], 1)
                    ),
                    dim=-1,
                )
            image_feature = image_feature.flatten(1, 2).transpose(0, 1)
            image_feature = torch.cat((base_image_feature, image_feature), dim=0)
        else:
            image_feature = image_feature[0]
            if self.image_newline is not None:
                image_feature = torch.cat((image_feature, self.image_newline[None]), dim=0)
        # import pdb; pdb.set_trace()
        # print(type(input))
        # import time
        # print("debug ERROR", image_feature.shape)
        # time.sleep(10)
        # import pdb; pdb.set_trace()
        # print(image_feature.shape)
        input['result'] = image_feature
        # import pdb; pdb.set_trace()
        

        
