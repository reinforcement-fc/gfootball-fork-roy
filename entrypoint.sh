#!/bin/bash

# Working directory 확인
echo "\n"
echo "Arguments passed to entrypoint.sh: $@"
echo "Current working directory: $(pwd)"

LOCAL_TAR_PATH="/opt/ml/code/sourcedir.tar.gz"

# source_dir을 다운로드 및 압축 해제
if [ ! -f "$LOCAL_TAR_PATH" ]; then
    echo "source_dir 파일이 존재하지 않습니다. S3에서 다운로드를 시작합니다."

    # 수동으로 S3에서 source_dir 압축 파일을 복사하여 압축 해제
    S3_SOURCE_PATH="s3://sagemaker-ap-southeast-2-443142193439/${TRAINING_JOB_NAME}/source/sourcedir.tar.gz"
    echo "Attempting to download source directory from S3 path: $S3_SOURCE_PATH"

    # S3에서 소스 파일 다운로드
    aws s3 cp "$S3_SOURCE_PATH" "$LOCAL_TAR_PATH"

    # 압축 해제
    if [ -f "$LOCAL_TAR_PATH" ]; then
        echo "Downloading is completed. extrac the file."
        tar -xzf "$LOCAL_TAR_PATH" -C /opt/ml/code
        echo "압축 해제 후 파일 목록:"
        ls -al /opt/ml/code
    else
        echo "다운로드에 실패하였습니다. 경로를 확인하십시오: $SAGEMAKER_SUBMIT_DIRECTORY"
        exit 1
    fi
else
    echo "source_dir 파일이 이미 존재합니다. 압축 해제를 건너뜁니다."
fi

# 현재 디렉토리 내 파일 목록 표시
echo "Final listing of files in /opt/ml/code:"
ls -al /opt/ml/code

python3 "$@"

# Python 스크립트가 완료된 후 /opt/ml 파일 목록 출력
echo "Files in /opt/ml after training completes:"
ls -R /opt/ml


# S3 업로드 경로 설정
S3_UPLOAD_PATH="s3://sagemaker-ap-southeast-2-443142193439/$TRAINING_JOB_NAME/opt_ml.tar.gz"

# 훈련 종료 후 /opt/ml 전체를 압축
echo "Compressing /opt/ml directory after training completes..."
tar -czf /opt/ml/opt_ml.tar.gz -C /opt ml

# S3에 업로드
echo "Uploading compressed /opt/ml to S3..."
aws s3 cp /opt/ml/opt_ml.tar.gz $S3_UPLOAD_PATH

if [ $? -ne 0 ]; then
    echo "Failed to upload to S3. Exiting."
    exit 1
fi

echo "Upload complete: $S3_UPLOAD_PATH"

