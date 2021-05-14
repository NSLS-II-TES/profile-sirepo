#!/bin/bash

pip install "bloptools>=0.0.2"

# TODO: start Sirepo Docker container here.

# https://github.com/NSLS-II/sirepo-bluesky/blob/4d7d556741a4a477966218dbdf80c3bb04ded404/.travis.yml#L52
# docker pull radiasoft/sirepo:beta

# https://github.com/NSLS-II/sirepo-bluesky/blob/4d7d556741a4a477966218dbdf80c3bb04ded404/.travis.yml#L59
# container_id=$(docker run -d -t --init --rm --name sirepo -e SIREPO_AUTH_METHODS=bluesky:guest -e SIREPO_AUTH_BLUESKY_SECRET=bluesky -e SIREPO_SRDB_ROOT=/sirepo -e SIREPO_COOKIE_IS_SECURE=false -p 8000:8000 -v $PWD/sirepo_bluesky/tests/SIREPO_SRDB_ROOT:/SIREPO_SRDB_ROOT:ro,z radiasoft/sirepo:beta bash -l -c "mkdir -v -p /sirepo/ && cp -Rv /SIREPO_SRDB_ROOT/* /sirepo/ && sirepo service http")
