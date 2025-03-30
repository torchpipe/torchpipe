import hami
import torchpipe
import torch
import sys, os
from typing import Tuple, List
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    
# from models.partial_hf import get_hf_model
from models import hf_helper
def get_page_address():
    pass
import hami
class PyPlugin:
    def init(self, params):
        self.params = params
        print(params)
        
        self.layer_idx = int(self.params['layer_idx'])

        self.addr = hami.result_wrapper(hami.init("TensorPage"))
        
        # self.addr_pool = hami._C.
    def forward(self, io: List[hami.Dict]):
        if self.layer_idx == 0:
            get_page_address()
            ## addr to tensor addr
        # print(list(io[0].keys()))
        input = io[0]['data']
        output = io[0]['output'][0]
        print(f"running: {self.layer_idx}, {output.shape}")
        # if self.params['layer_idx'] == '0':
        #     print("input[1] = ", input[1])
        # print([x.shape for x in input])
        # print([x.shape for x in output])



if __name__ == '__main__':
    import time
    # time.sleep(10)

    hami.register("TorchPlugin", PyPlugin)
    
    
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
    io = {'data':input_ids.squeeze(0),"node_name":'embed_token'}
    #     id_type req_id;
    # int32_t req_tokens{0};
    # int32_t new_tokens{0};
    # int32_t max_new_tokens{0};
    # int32_t max_tokens{0};
    io[hami.TASK_REQUEST_ID_KEY] = "id"
    io[hami.TASK_MSG_KEY] = hami.TypedDict({"req_tokens": 5,
                                            "new_tokens": 0,
                                            "max_new_tokens": 7,
                                            "max_tokens":4096})
    model(io)
    print([x.shape for x in io['result']])
    print(io['result'])