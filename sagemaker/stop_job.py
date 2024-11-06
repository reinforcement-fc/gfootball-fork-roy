import boto3
import argparse


def stop_training_job(training_job_name):
    # Initialize the SageMaker client
    sagemaker_client = boto3.client("sagemaker")


    # Stop the training job
    sagemaker_client.stop_training_job(TrainingJobName=training_job_name)
    print(f"Stopped SageMaker training job: {training_job_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stop SageMaker Training Job")
    parser.add_argument("training_job_name", type=str, help="The name of the SageMaker training job to monitor")
    args = parser.parse_args()

    # 훈련 작업 모니터링 시작
    stop_training_job(args.training_job_name)
