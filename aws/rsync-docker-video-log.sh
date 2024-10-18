#!/bin/bash

source /path/to/config.sh

rsync -avz -e "ssh -i $SSH_KEY_PATH" ec2-user@$EC2_IP:/tmp/docker_* ./log_from_ec2/
