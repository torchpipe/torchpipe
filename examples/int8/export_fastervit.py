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

import faster_vit 
import torch


def export_onnx(torch_model, input_shape, onnx_save_path):
    x = torch.randn(1,*input_shape)
    out_size = 1
    out={"input":{0:"batch_size"}}
    for i in range(out_size):
        out[f"output_{i}"] = {0:"batch_size"}

    torch_model.eval()
    torch.onnx.export(torch_model,
                        x,
                        onnx_save_path,
                        opset_version=17,
                        do_constant_folding=True,   
                        input_names=["input"],      # 输入名
                        output_names=[f"output_{i}" for i in range(out_size)],  # 输出名
                        dynamic_axes=out)
    import onnx
    from onnxsim import onnx_simplifier

    model_simp, check = onnx_simplifier.simplify(onnx_save_path, check_n = 0)
    model_simp = onnx.shape_inference.infer_shapes(model_simp)
    onnx.save(model_simp, onnx_save_path)
    print(f"{onnx_save_path} saved.")

if __name__ == "__main__":   
    # Define fastervit-0 model with 224 x 224 resolution
    from timm.models import create_model
    model = create_model(
        'faster_vit_0_224',
        num_classes=3)
    # load Pretrained Models from https://github.com/NVlabs/FasterViT/tree/main
    
    export_onnx(model, (3,224,224), "faster_vit_0_224.onnx")