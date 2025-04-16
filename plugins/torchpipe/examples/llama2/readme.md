### paramater export 
```bash
# python models/hf_helper.py 

# export HF_ENDPOINT=https://hf-mirror.com
python models/export_onnx_v2.py --num_layers=2
```

## run
```
python -c "import torch; print(torch.__version__, torch.version.cuda)"
pip install flashinfer-python -i https://flashinfer.ai/whl/cu128/torch2.7


python plain_llama2.py
# num_layers = 2:
# San Francisco is a totalitéaletoreignersbyMSран 
```

## streaming
```
python chat_client.py --prompt="San Francisco is a" --max_tokens 7

python chat_client.py --prompt="Do you know the book Traction by Gino Wickman? generate anwser >= 2000 wordsDo you know the book Traction by Gino Wickman? generate anwser >= 2000 wordsDo you know the book Traction by Gino Wickman? generate anwser >= 2000 wordsDo you know the book Traction by Gino Wickman? generate anwser >= 2000 wordsDo you know the book Traction by Gino Wickman? generate anwser >= 2000 wordsDo you know the book Traction by Gino Wickman? generate anwser >= 2000 wordsDo you know the book Traction by Gino Wickman? generate anwser >= 2000 wordsDo you know the book Traction by Gino Wickman? generate anwser >= 2000 wordsDo you know the book Traction by Gino Wickman? generate anwser >= 2000 wordsDo you know the book Traction by Gino Wickman? generate anwser >= 2000 wordsDo you know the book Traction by Gino Wickman? generate anwser >= 2000 words" --max_tokens 2048 

python chat_client.py --prompt="Do you know the book Traction by Gino Wickman?" --max_tokens 132  

```


<!-- https://github.com/dreaming-panda/MagicEnc/blob/0d05cec01cdff53d51daa7402fa267595e3bc12b/llama.py#L66 -->