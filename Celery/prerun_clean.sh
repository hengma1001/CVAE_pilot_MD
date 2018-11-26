#!/bin/bash

echo -e 'This will kill all the celery workers and python programs on GPUs'

rabbitmqctl stop
ps aux | grep 'celery worker' | awk '{print $2}' | xargs kill -9 
# nvidia-smi | grep 'python'  | awk '{print $3}' | xargs kill -9 

