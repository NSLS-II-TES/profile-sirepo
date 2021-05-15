#!/bin/bash

# pip install "bloptools>=0.0.2" --no-deps -vv

conda install -y -c ${CONDA_CHANNEL_NAME} "bloptools>=0.0.2"

# Start Sirepo Docker container.
# https://github.com/NSLS-II/sirepo-bluesky/blob/4d7d556741a4a477966218dbdf80c3bb04ded404/.travis.yml#L59

git clone https://github.com/NSLS-II/sirepo-bluesky

SIREPO_DOCKER_IMAGE="radiasoft/sirepo:beta"

docker pull ${SIREPO_DOCKER_IMAGE}

docker run -d -t --init --rm --name sirepo \
    -e SIREPO_AUTH_METHODS=bluesky:guest \
    -e SIREPO_AUTH_BLUESKY_SECRET=bluesky \
    -e SIREPO_SRDB_ROOT=/sirepo \
    -e SIREPO_COOKIE_IS_SECURE=false \
    -p 8000:8000 \
    -v $PWD/sirepo-bluesky/sirepo_bluesky/tests/SIREPO_SRDB_ROOT:/SIREPO_SRDB_ROOT:ro,z \
    ${SIREPO_DOCKER_IMAGE} \
    bash -l -c "mkdir -v -p /sirepo/ && cp -Rv /SIREPO_SRDB_ROOT/* /sirepo/ && sirepo service http"

docker images
docker ps -a
