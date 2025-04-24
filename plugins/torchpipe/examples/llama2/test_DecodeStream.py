from tokenizers.decoders import ByteFallback, DecodeStream
from tokenizers.models import BPE, Model, Unigram
from tokenizers import AddedToken, Encoding, Tokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import tokenizers
import torch

class TestTokenizer:
    def test_decode(self):
        tokenizer = Tokenizer(BPE())
        tokenizer.add_tokens(["my", "name", "is", "john", "pair"])

        # Can decode single sequences
        output = tokenizer.decode([0, 1, 2, 3])
        assert output == "my name is john"

        # Can decode batch
        output = tokenizer.decode_batch([[0, 1, 2, 3], [4]])
        assert output == ["my name is john", "pair"]

        # Can decode stream
        stream = DecodeStream(skip_special_tokens=True)
        assert stream.step(tokenizer, 0) == "my"
        assert stream.step(tokenizer, 1) == " name"
        assert stream.step(tokenizer, 2) == " is"
        assert stream.step(tokenizer, 3) == " john"
        
    def test_llama(self):
        exported_params = "./exported_params"
        self.old_tokenizer = AutoTokenizer.from_pretrained(exported_params)
        self.tokenizer=tokenizers.Tokenizer.from_file(exported_params + "/tokenizer.json")
        eos=self.tokenizer.get_vocab()["</s>"]
        print(f'self.tokenizer.eos_token_id={eos}')
        
        # print(self.tokenizer.get_vocab())
        # self.tokenizer=tokenizers.Tokenizer.from_pretrained(exported_params)
        # print(self.tokenizer.__class__.__name__)
        print(self.tokenizer.encode("my name is john pair").ids)
        input_ids = self.tokenizer.encode("my name is john pair").ids
        print(f'input_ids={input_ids};\n')
        # self.tokenizer.add_tokens(["my", "name", "is", "john", "pair"])
        output = self.tokenizer.decode([12232,33121,22222, 29871])
        print(f'output={output};\n')
        special = self.tokenizer.decode([29871])
        print(f'special: {special}:{len(special)}')
        print(self.tokenizer.id_to_token(29871))
        zz = [13, 13, 5634, 13, 13, 243, 162, 151, 168, 13, 13, 12148, 1653, 385, 13955, 29954, 8607, 411, 263, 11855, 310, 29871, 29945, 29900, 29995, 310, 278, 1776, 637, 2920, 29892, 10423, 411, 263, 2654, 2927, 29889, 2]
        print(self.old_tokenizer.decode(zz, skip_special_tokens=True))
        stream = DecodeStream(skip_special_tokens=True)
        print('xxxx'*4)
        index = 0
        for id in zz:
            re = stream.step(self.tokenizer, id)
            if re is None:
                print(f'{index}: None|{id}|{self.tokenizer.id_to_token(id)}')
            else:
                print(f'{index}: {re}|{len(re)}|{id}')
            index+=1
        
        
        # print(stream.step(self.tokenizer, 2))
        # zz=self.old_tokenizer.decode(2, skip_special_tokens=True)
        # print(zz, type(zz) ,zz=="")
if __name__ == "__main__":
    a = TestTokenizer()
    a.test_llama()
    
    from torch import tensor
    
    # a=[tensor([13]), tensor([13]), tensor([5634]), tensor([13]), tensor([13]), tensor([243]), tensor([162]), tensor([151]), tensor([168]), tensor([13]), tensor([13]), tensor([12148]), tensor([1653]), tensor([385]), tensor([13955]), tensor([29954]), tensor([8607]), tensor([411]), tensor([263]), tensor([11855]), tensor([310]), tensor([29871]), tensor([29945]), tensor([29900]), tensor([29995]), tensor([310]), tensor([278]), tensor([1776]), tensor([637]), tensor([2920]), tensor([29892]), tensor([10423]), tensor([411]), tensor([263]), tensor([2654]), tensor([2927]), tensor([29889]), tensor([2])]
    # b = torch.cat(a)
    # print(b.tolist())