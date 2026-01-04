


docker run --rm  --name omniback2014 --gpus=all --ipc=host --network=host -v $(pwd):/workspace --shm-size 1G --ulimit memlock=-1 --ulimit stack=67108864  --privileged=true  -w /workspace \
 -it omniback:2014  bash
 



docker run --rm --name omniback228  --gpus=all --ipc=host --network=host -v $(pwd):/workspace --shm-size 1G --ulimit memlock=-1 --ulimit stack=67108864  --privileged=true  -w /workspace \
 -it omniback:228  bash


#   sh docker/common/install_python.sh 3.10