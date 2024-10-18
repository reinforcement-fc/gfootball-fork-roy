#!/bin/bash
#
source /path/to/config.sh

ssh -L 5901:localhost:5901 -i $SSH_KEY_PATH ec2-user@$EC2_IP
