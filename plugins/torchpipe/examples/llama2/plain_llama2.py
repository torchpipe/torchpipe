import hami
import torchpipe
import torch
import sys, os
from typing import Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    
# from models.partial_hf import get_hf_model
from models import hf_helper



def register_trt_plugin():
    import tensorrt.plugin as trtp
    import numpy.typing as npt
    import numpy as np

    @trtp.register("CustomTorchOps::TorchPlugin")
    def circ_pad_plugin_desc(
        inp0: trtp.TensorDesc, pads: npt.NDArray[np.int32]
    ) -> trtp.TensorDesc:
        ndim = inp0.ndim
        out_desc = inp0.like()

        for i in range(np.size(pads) // 2):
            out_desc.shape_expr[ndim - i - 1] += int(
                pads[i * 2] + pads[i * 2 + 1]
            )

        return out_desc
    
    @trtp.impl("CustomTorchOps::TorchPlugin")
    def circ_pad_plugin_impl(
        inp0: trtp.Tensor,
        pads: npt.NDArray[np.int32],
        outputs: Tuple[trtp.Tensor],
        stream: int
    ) -> None:
        inp_t = torch.as_tensor(inp0, device="cuda")
        out_t = torch.as_tensor(outputs[0], device="cuda")

        out = torch.nn.functional.pad(inp_t, pads.tolist(), mode="circular")
        out_t.copy_(out)
if __name__ == '__main__':
    register_trt_plugin()
    
    
    model = hami.init_from_file('config/plain_llama2.toml')
    
    exported_params = "./exported_params"
    tokenizer = AutoTokenizer.from_pretrained(exported_params)
    
    # inference
    prompt = "San Francisco is a"
    inputs = tokenizer(prompt, return_tensors="pt", return_attention_mask=False)
    input_ids = inputs['input_ids']
    # attention_mask = inputs['attention_mask']
    print(inputs, input_ids.shape)
    # print(io)
    io = {'data':input_ids.squeeze(0)}
    model(io)
    print([x.shape for x in io['result']])
    print(io['result'][1])