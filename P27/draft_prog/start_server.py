from tasks import run_omm_on_gpu 
import numpy as np
import time
import GPUtil
import os 
import subprocess
import h5py 

from utils import start_worker, start_flower_monitor, omm_job
from tasks import app

start_worker(app) 
# start_flower_monitor()

print 'test'