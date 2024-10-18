#!/bin/bash

# Define the levels with low difficulty
LOW_DIFFICULTY_LEVELS=(
  "academy_empty_goal_close"
  "academy_empty_goal"
  "academy_run_to_score"
  "academy_run_with_ball"
  "academy_pass"
)

# Loop through each level and call the docker_test_with_video.sh script
for LEVEL in "${LOW_DIFFICULTY_LEVELS[@]}"
do
  echo "Running level: $LEVEL"

  # Set LEVEL as an environment variable and call the script
  export LEVEL="$LEVEL"
  ./docker_test_with_video.sh

done
