#!/bin/bash
set -e


#docker build --build-arg DOCKER_BASE=tensorflow/tensorflow:1.15.5-gpu-py3-jupyter . -t gfootball_docker_test
#docker run --gpus all -e DISPLAY=:1 -e XAUTHORITY=/root/.Xauthority -v /tmp/.X11-unix:/tmp/.X11-unix:rw -v $HOME/.Xauthority:/root/.Xauthority:rw --entrypoint bash -it gfootball_docker_test -c 'set -e; for x in `find gfootball/env -name *_test.py`; do UNITTEST_IN_DOCKER=1 PYTHONPATH=/ python3 $x; done'

#docker build --build-arg DOCKER_BASE=tensorflow/tensorflow:1.15.5-gpu-py3-jupyter  . -t gfootball_docker_test -f Dockerfile_examples
docker run --gpus all -e DISPLAY=:1 -e XAUTHORITY=/root/.Xauthority -v /tmp/.X11-unix:/tmp/.X11-unix:rw -v $HOME/.Xauthority:/root/.Xauthority:rw --entrypoint python3 -it gfootball_docker_test gfootball/examples/run_ppo2.py --level=academy_empty_goal_close --num_timesteps=100000

echo "Test successful!!!"
