

# Experimental
```
pip install pynvml matplotlib
CUDA_VISIBLES_DEVICES=1 python resnet50.py
```


```
docker run --rm   --network=host --gpus=all  --ulimit memlock=-1 --ulimit stack=67108864  --privileged=true  -v `pwd`:/workspace -it base_trt-9.1  /bin/bash 
 
 ```