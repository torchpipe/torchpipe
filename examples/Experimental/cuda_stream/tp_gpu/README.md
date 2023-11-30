


```
pip install thrift

# apt-get install thrift-compiler -y
# thrift -r -out ./ --gen py serve.thrift

python3 start.py

python3 client.py
```


```
python3 resnet50.py --config=resnet50_gpu_decode.toml 
```

## test multiple processes
```
python3 resnet50.py --config=resnet50_gpu_decode_half.toml 
python3 resnet50.py --config=resnet50_gpu_decode_half.toml 

```


## test pipeline

```
python3 resnet50.py --config=resnet50_gpu_decode_dual.toml 
 
```