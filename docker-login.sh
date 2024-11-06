#!/bin/bash

#run next command in 443142193439 shell.
#`aws ecr get-login-password --region ap-southeast-2`
PASSWORD=FROM_443142193439_ACCOUNT

echo $PASSWORD | docker login --username AWS --password-stdin 443142193439.dkr.ecr.ap-southeast-2.amazonaws.com
