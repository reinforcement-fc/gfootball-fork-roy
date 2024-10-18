#!/bin/bash

docker run -e DISPLAY=:1 -e XAUTHORITY=/root/.Xauthority -v /tmp/.X11-unix:/tmp/.X11-unix:rw -v $HOME/.Xauthority:/root/.Xauthority:rw --gpus all --entrypoint /bin/bash -it gfootball_docker_test
