set -e
docker build -f docker/Dockerfile.manylinux2_28 -t omniback:228 docker/
# om::228

docker build -f docker/Dockerfile.manylinux2014 -t omniback:2014 docker/
# om::2014

