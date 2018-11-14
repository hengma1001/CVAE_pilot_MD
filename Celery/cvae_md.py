from __future__ import print_function

print('Running the MD simulation according to the CVAE classification ')

from glob import glob
import numpy as np
import sys, os, h5py, time, errno
import GPUtil

from utils import start_rabbit, start_worker, start_flower_monitor, read_h5py_file, cm_to_cvae, job_on_gpu
from utils import omm_job, cvae_job 



GPU_ids = [gpu.id for gpu in GPUtil.getGPUs()] 
print('Available GPUs', GPU_ids) 

top_file = os.path.abspath('../P27-all/C1B48/C1B48.top.gz')
pdb_file = os.path.abspath('../P27-all/C1B48/C1B48.pdb.gz')

# number of cvae jobs 
n_cvae = 1 

work_dir = os.path.abspath('./')


# Starting RabbitMQ and Celery schedule framework. 
log_dir = os.path.join(work_dir, 'scheduler_logs') 

try:
    os.mkdir(log_dir)
except OSError as exc:
    if exc.errno != errno.EEXIST:
        raise
    pass

rabbitmq_log = os.path.join(log_dir, 'rabbit_server_log.txt') 
start_rabbit(rabbitmq_log)
time.sleep(5)

celery_worker_log = os.path.join(log_dir, 'celery_worker_log.txt') 
start_worker(celery_worker_log)
start_flower_monitor() 
print('Waiting 10 seconds for the server to stablize.')
time.sleep(10)


# Starting MD simulation on all GPUs using OpenMM 
jobs = []
for gpu_id in GPU_ids: 
    job = omm_job(job_id=int(time.time()), gpu_id=gpu_id, top_file=top_file, pdb_file=pdb_file)
    job.start() 
    jobs.append(job) 
    print('Started OpenMM jobs on GPU', gpu_id)
    time.sleep(2)
    
print('Waiting 5 mins for omm to write valid contact map .h5 files ')
# time.sleep(300) 


# Read all the contact map .h5 file in local dir
cm_files = glob('*/*_cm.h5')
cm_data_lists = [read_h5py_file(cm_file) for cm_file in cm_files] 

print('Waiting for the OpenMM to complete the first 100,000 frames as CVAE training data')
frame_number = lambda lists: sum([cm.shape[1] for cm in lists]) 

frame_marker = 0 
# change to 1e5 later, HM
while frame_number(cm_data_lists) < 500: #100000: 
    for cm in cm_data_lists: 
        cm.refresh() 
    if frame_number(cm_data_lists) > frame_marker: 
        print('Current number of frames from OpenMM:', frame_number(cm_data_lists)) 
        frame_marker += int((10000 + frame_marker) / 10000) * 10000
        print('    Next report at frame', frame_marker) 

print('Ready for CAVE with total number of frames:', frame_number(cm_data_lists)) 

# Compress all .h5 files into one 
cvae_input = cm_to_cvae(cm_data_lists) 
train_data_length = [ cm_data.shape[1] for cm_data in cm_data_lists]

# Write the traj info 
omm_log = os.path.join(log_dir, 'openmm_log.txt') 
log = open(omm_log, 'w') 
for i, n_frame in enumerate(train_data_length): 
    log.writelines("{} {}\n".format(cm_files[0], n_frame))    
log.close()


cvae_input_dir = os.path.join(work_dir, 'cvae_input') 
try:
    os.mkdir(cvae_input_dir)
except OSError as exc:
    if exc.errno != errno.EEXIST:
        raise
    pass

cvae_input_file = os.path.join(cvae_input_dir, 'cvae_input.h5')
cvae_input_save = h5py.File(cvae_input_file, 'w')
cvae_input_save.create_dataset('contact_maps', data=cvae_input)
cvae_input_save.close() 

# CVAE
hyper_dims = np.array(range(n_cvae)) + 3
print('Running CVAE for hyper dimension:', hyper_dims) 

for i in range(n_cvae): 
    cvae_j = cvae_job(time.time(), i, cvae_input_file, hyper_dim=3) 
    stop_jobs = job_on_gpu(i, jobs) 
    stop_jobs.stop()
    jobs.remove(stop_jobs) 
    cvae_j.start() 
    jobs.append(cvae_j) 