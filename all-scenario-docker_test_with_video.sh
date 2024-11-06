#!/bin/bash

echo "Starting at $(date)"

while IFS= read -r scenario; do
    # LEVEL 변수 설정 및 스크립트 실행
    echo "LEVEL=$scenario ./docker_test_with_video.sh"
    LEVEL=$scenario ./docker_test_with_video.sh
done < scenarios.txt

echo "Done at $(date)"
