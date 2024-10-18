#!/bin/bash


# Get the current date and time in the format YYYY-MM-DD_HH-MM-SS
current_time=$(date +"%Y-%m-%d_%H-%M-%S")

DOCKER_VIDEO_FILE="/tmp/docker_${LEVEL}_video_${current_time}.mp4"
DOCKER_LOG_FILE="/tmp/docker_${LEVEL}_log_${current_time}.txt"

# Start ffmpeg in the background and store its PID
ffmpeg -f x11grab -s 1024x768 -i :1 -r 30 -vcodec libx264 -preset fast -pix_fmt yuv420p "$DOCKER_VIDEO_FILE" &
FFMPEG_PID=$!

NUM_TIMESTEPS=10000 LOG_FILE=$DOCKER_LOG_FILE ./docker_test.sh

# Once the Docker process completes, terminate ffmpeg
kill $FFMPEG_PID

ls -hl "$DOCKER_VIDEO_FILE" "$DOCKER_LOG_FILE"

echo "DONE."

