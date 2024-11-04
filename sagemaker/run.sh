#!/bin/bash


export TRAINING_JOB_NAME="gfootball-training-job-$(date +'%Y-%m-%d-%H-%M-%S')"
export INSTANCE_TYPE="ml.g4dn.2xlarge"
#export INSTANCE_TYPE="ml.m5.large"

echo $TRAINING_JOB_NAME 

python train_gfootball_on_custom_docker.py "$@"

if [ $? -ne 0 ]; then
    echo "Sagamaker Python script failed. Exiting."
    exit 1
fi


MODEL_S3_PATH="s3://sagemaker-ap-southeast-2-443142193439/$TRAINING_JOB_NAME/output/model.tar.gz"
LOG_S3_PATH="s3://sagemaker-ap-southeast-2-443142193439/$TRAINING_JOB_NAME/output/output.tar.gz"

# 로컬 저장 경로 설정
LOCAL_MODEL_PATH="./downloaded_model"
LOCAL_LOG_PATH="./downloaded_logs"

# 모델 및 로그 파일 다운로드 디렉토리 생성
mkdir -p $LOCAL_MODEL_PATH
mkdir -p $LOCAL_LOG_PATH

echo "Downloading model artifacts from S3..."
aws s3 cp $MODEL_S3_PATH $LOCAL_MODEL_PATH/model.tar.gz

if [ $? -ne 0 ]; then
    echo "aws s3 cp failed. Exiting."
    exit 1
fi

echo "Downloading log files from S3..."
aws s3 cp $LOG_S3_PATH $LOCAL_LOG_PATH/output.tar.gz

if [ $? -ne 0 ]; then
    echo "aws s3 cp failed. Exiting."
    exit 1
fi

# 모델 압축 해제
tar -xzf $LOCAL_MODEL_PATH/model.tar.gz -C $LOCAL_MODEL_PATH

# 로그 압축 해제
tar -xzf $LOCAL_LOG_PATH/output.tar.gz -C $LOCAL_LOG_PATH

echo "Model and logs have been downloaded and extracted."
