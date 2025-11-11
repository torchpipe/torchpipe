set -e
docker build -f docker/Dockerfile.manylinux2_28 -t omniback:228 docker/
# omniback::228

docker build -f docker/Dockerfile.manylinux2014 -t omniback:2014 docker/
# omniback::2014

