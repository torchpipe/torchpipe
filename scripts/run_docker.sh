


docker run --rm  --name hami2014 --gpus=all --ipc=host --network=host -v $(pwd):/workspace --shm-size 1G --ulimit memlock=-1 --ulimit stack=67108864  --privileged=true  -w /workspace \
 -it hami:2014  bash
 



docker run --rm --name hami228  --gpus=all --ipc=host --network=host -v $(pwd):/workspace --shm-size 1G --ulimit memlock=-1 --ulimit stack=67108864  --privileged=true  -w /workspace \
 -it hami:228  bash


#   sh docker/common/install_python.sh 3.10