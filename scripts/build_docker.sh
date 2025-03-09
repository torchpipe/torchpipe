set -e
docker build -f docker/Dockerfile.manylinux2_28 -t hami:228 docker/
# hami::228

docker build -f docker/Dockerfile.manylinux2014 -t hami:2014 docker/
# hami::2014

