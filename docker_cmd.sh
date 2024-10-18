docker build --build-arg DOCKER_BASE=tensorflow/tensorflow:1.15.5-gpu-py3-jupyter -t gfootball .
docker run --gpus all -it   -e DISPLAY=:1   -e XAUTHORITY=/root/.Xauthority   -v /tmp/.X11-unix/X1:/tmp/.X11-unix/X1   -v $HOME/.Xauthority:/root/.Xauthority:rw   gfootball bash

