#!/bin/bash


export TRAINING_JOB_NAME="gfootball-training-job-$(date +'%Y-%m-%d-%H-%M-%S')"
export INSTANCE_TYPE="ml.g4dn.2xlarge"
#export INSTANCE_TYPE="ml.m5.large"

echo $TRAINING_JOB_NAME 

python train_gfootball_on_custom_docker.py "$@"

if [ $? -ne 0 ]; then
    echo "Training script failed. Exiting."
    exit 1
fi

OPT_ML_S3_PATH="s3://sagemaker-ap-southeast-2-443142193439/$TRAINING_JOB_NAME/opt_ml.tar.gz"
LOCAL_MODEL_PATH="./downloaded_model/$TRAINING_JOB_NAME"

echo "if somthing go wrong use this cli for yourself."
echo ">   aws s3 cp $OPT_ML_S3_PATH $LOCAL_MODEL_PATH/opt_ml.tar.gz"
mkdir -p $LOCAL_MODEL_PATH


python monitor_job.py "$TRAINING_JOB_NAME"

if [ $? -ne 0 ]; then
    echo "Monitoring job script failed. Exiting."
    exit 1
fi


echo "Downloading model artifacts from S3..."
echo "aws s3 cp $OPT_ML_S3_PATH $LOCAL_MODEL_PATH/opt_ml.tar.gz"
aws s3 cp $OPT_ML_S3_PATH $LOCAL_MODEL_PATH/opt_ml.tar.gz

if [ $? -ne 0 ]; then
    echo "aws s3 cp failed. Exiting."
    echo "python monitor_job $TRAINING_JOB_NAME"
    exit 1
fi

# 모델 압축 해제
tar -xzf $LOCAL_MODEL_PATH/opt_ml.tar.gz -C $LOCAL_MODEL_PATH


if [ $? -ne 0 ]; then
    echo "tar xzf failed. Exiting."
    exit 1
fi


echo "Model and logs have been downloaded and extracted."
