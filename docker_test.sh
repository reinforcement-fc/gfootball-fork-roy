#!/bin/bash

# Use provided environment variables or default values
NUM_TIMESTEPS=${NUM_TIMESTEPS:-1000}
LEVEL=${LEVEL:-academy_empty_goal_close}
POLICY=${POLICY:-cnn}
LOG_FILE=${LOG_FILE:-/tmp/docker-log.txt}

# Define a log file
#
echo $NUM_TIMESTEPS

start_time=$(date +%s)

docker run -e DISPLAY=:1 -e XAUTHORITY=/root/.Xauthority -v /tmp/.X11-unix:/tmp/.X11-unix:rw -v $HOME/.Xauthority:/root/.Xauthority:rw --gpus all --entrypoint python3 -it gfootball_docker_test gfootball/examples/run_ppo2.py --level=$LEVEL --num_timesteps=$NUM_TIMESTEPS --policy=$POLICY --render > "$LOG_FILE" 2>&1

end_time=$(date +%s)

# Calculate the duration
elapsed_time=$(( end_time - start_time ))

echo "Execution time: $elapsed_time seconds" | tee -a "$LOG_FILE"
