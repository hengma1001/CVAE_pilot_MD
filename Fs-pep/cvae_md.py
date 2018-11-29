from __future__ import print_function
print('============================================================== ')
print('Running the MD simulation according to the CVAE classification ')
print('============================================================== ')

from glob import glob
import numpy as np
import sys, os, h5py, time, errno
# import GPUtil, subprocess32
import subprocess32
from sklearn.cluster import DBSCAN

from utils import start_rabbit, start_worker, start_flower_monitor, read_h5py_file, cm_to_cvae, job_on_gpu
from utils import find_frame, write_pdb_frame, make_dir_p, job_list, outliers_from_cvae
from utils import omm_job, cvae_job 

from CVAE import CVAE

n_gpus = 16
GPU_ids = range(n_gpus) # [gpu.id for gpu in GPUtil.getGPUs()] 
print('Available GPUs', GPU_ids) 

os.environ["RABBITMQ_MNESIA_BASE"] = "~/.rabbit_base"
os.environ["RABBITMQ_LOG_BASE"] = "~/.rabbit_base/"

# top_file = os.path.abspath('../P27-all/C1B48/C1B48.top.gz')
# pdb_file = os.path.abspath('../P27-all/C1B48/C1B48.pdb.gz')
top_file = None
pdb_file = os.path.abspath('./pdb/100-fs-peptide-400K.pdb')

# number of cvae jobs, from hyper_dim 3 
n_cvae = 4 

work_dir = os.path.abspath('./')

# create folders for store results
log_dir = os.path.join(work_dir, 'scheduler_logs') 
make_dir_p(log_dir)

# Starting RabbitMQ and Celery schedule framework. 
rabbitmq_log = os.path.join(log_dir, 'rabbit_server_log.txt') 
start_rabbit(rabbitmq_log)
time.sleep(5)

celery_worker_log = os.path.join(log_dir, 'celery_worker_log.txt') 
start_worker(celery_worker_log)
# start_flower_monitor() 
print('Waiting 10 seconds for the server to stablize.')
time.sleep(10)


# Starting MD simulation on all GPUs using OpenMM 
jobs = job_list() 
for gpu_id in GPU_ids: 
    job = omm_job(job_id=int(time.time()), gpu_id=gpu_id, top_file=top_file, pdb_file=pdb_file)
    job.start() 
    jobs.append(job) 
    print('Started OpenMM jobs on GPU', gpu_id)
    time.sleep(2)
    
print('Waiting 5 mins for omm to write valid contact map .h5 files ')
time.sleep(120) 


# Read all the contact map .h5 file in local dir
cm_files = sorted(glob('./omm*/*_cm.h5'))
cm_data_lists = [read_h5py_file(cm_file) for cm_file in cm_files] 

print('Waiting for the OpenMM to complete the first 100,000 frames as CVAE training data')
frame_number = lambda lists: sum([cm.shape[1] for cm in lists]) 

frame_marker = 0 
# number of training frames for cvae, change to 1e5 later, HM 
while frame_number(cm_data_lists) < 20000: 
    for cm in cm_data_lists: 
        cm.refresh() 
    if frame_number(cm_data_lists) >= frame_marker: 
        print('Current number of frames from OpenMM:', frame_number(cm_data_lists)) 
        frame_marker = int((10000 + frame_marker) / 10000) * 10000
        print('    Next report at frame', frame_marker) 

print('Ready for CAVE with total number of frames:', frame_number(cm_data_lists)) 

# Compress all .h5 files into one in cvae format 
cvae_input = cm_to_cvae(cm_data_lists) 
train_data_length = [cm_data.shape[1] for cm_data in cm_data_lists]

# Write the traj info 
omm_log = os.path.join(log_dir, 'openmm_log.txt') 
log = open(omm_log, 'w') 
for i, n_frame in enumerate(train_data_length): 
    log.writelines("{} {}\n".format(cm_files[i], n_frame))    
log.close()

# Create .h5 input for cvae
cvae_input_dir = os.path.join(work_dir, 'cvae_input') 
make_dir_p(cvae_input_dir)

cvae_input_file = os.path.join(cvae_input_dir, 'cvae_input.h5')
cvae_input_save = h5py.File(cvae_input_file, 'w')
cvae_input_save.create_dataset('contact_maps', data=cvae_input)
cvae_input_save.close() 

# CVAE
hyper_dims = np.array(range(n_cvae)) + 3
print('Running CVAE for hyper dimension:', hyper_dims) 

for i in range(n_cvae): 
    cvae_j = cvae_job(time.time(), i, cvae_input_file, hyper_dim=hyper_dims[i]) 
    stop_jobs = jobs.get_job_from_gpu_id(i) 
    stop_jobs.stop()  
    print('Started CVAE for hyper dimension:', hyper_dims[i])
    time.sleep(2)
    cvae_j.start() 
    jobs.append(cvae_j) 
    time.sleep(2)

while [cvae_j.job.status for cvae_j in jobs.get_cvae_jobs()] != [u'SUCCESS'] * len(jobs.get_cvae_jobs()): 
    time.sleep(1)
print('CVAE jobs done. ') 

for cvae_j in jobs.get_cvae_jobs(): 
    cvae_j.stop()

# All the outliers from cvae
outlier_list = []
for cvae_j in jobs.get_cvae_jobs(): 
    outliers = outliers_from_cvae(cvae_j.job.result[0], cvae_input, hyper_dim=cvae_j.hyper_dim, eps=0.35) 
    outlier_list.append(outliers) 
    
outlier_list = np.unique(np.array(outlier_list).flatten()) 

# write the pdb according the outlier indices
traj_info = open('./scheduler_logs/openmm_log.txt', 'r').read().split()

traj_dict = dict(zip(traj_info[::2], np.array(traj_info[1::2]).astype(int)))

outliers_pdb = os.path.join(work_dir, 'outliers_pdb')
make_dir_p(outliers_pdb)

for outlier in outlier_list: 
    traj_file, frame_number = find_frame(traj_dict, outlier) 
    outlier_pdb_file = os.path.join(outliers_pdb, '%d.pdb' % outlier)
    outlier_pdb = write_pdb_frame(traj_file, pdb_file, frame_number, outlier_pdb_file) 
    
print('Finishing and cleaning up the jobs. ')
subprocess.Popen('bash prerun_clean.sh'.split(" "))
