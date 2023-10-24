# git clone -b 0.3.2b2 ssh://git@g.hz.netease.com:22222/deploy/torchpipe.git
# cd torchpipe/ && git submodule update --init --recursive




img_name=nvcr.io/nvidia/pytorch:22.12-py3 #  for cuda11
# img_name=nvcr.io/nvidia/pytorch:23.05-py3 #  for tensort8.6.1, LayerNorm
docker run --rm --gpus=all --ipc=host  --network=host -v `pwd`:/workspace  --shm-size 1G  --ulimit memlock=-1 --ulimit stack=67108864  --privileged=true -it $img_name /bin/bash

# python setup.py bdist_wheel && ll dist/ -h
