#!/bin/bash
set -e


# for just unit test
#docker build --build-arg DOCKER_BASE=tensorflow/tensorflow:1.15.5-gpu-py3-jupyter . -t gfootball_docker_test
#docker run --gpus all -e DISPLAY=:1 -e XAUTHORITY=/root/.Xauthority -v /tmp/.X11-unix:/tmp/.X11-unix:rw -v $HOME/.Xauthority:/root/.Xauthority:rw --entrypoint bash -it gfootball_docker_test -c 'set -e; for x in `find gfootball/env -name *_test.py`; do UNITTEST_IN_DOCKER=1 PYTHONPATH=/ python3 $x; done'

VERSION=2.7

# for traning.
docker build --build-arg DOCKER_BASE=tensorflow/tensorflow:1.15.5-gpu-py3-jupyter . -t gfootball-fork:$VERSION -f Dockerfile_sagemaker
docker tag gfootball-fork:$VERSION 443142193439.dkr.ecr.ap-southeast-2.amazonaws.com/gfootball-fork:$VERSION
docker push 443142193439.dkr.ecr.ap-southeast-2.amazonaws.com/gfootball-fork:$VERSION
