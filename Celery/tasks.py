from celery import Celery
import simtk.unit as u 
from run_cvae import run_cvae
import sys, os
import errno 

sys.path.append('/home/hm0/Research/molecules/molecules_git/build/lib')
from molecules.sim.openmm_simulation import openmm_simulate_charmm_nvt

app = Celery('tasks', broker='pyamqp://guest@localhost//', backend='rpc://') 



@app.task
def add(x, y):
    return x + y

@app.task
def run_omm_with_celery(run_id, gpu_index, top_file, pdb_file, check_point=None): 
    work_dir = os.getcwd()
    iter_dir = os.path.join(work_dir, "omm_run%d" % int(run_id))
    
    try:
        os.mkdir(iter_dir)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass
    
    os.chdir(iter_dir) 

    openmm_simulate_charmm_nvt(top_file, pdb_file, 
                               check_point = check_point, 
                               GPU_index=gpu_index, 
                               output_traj="output.dcd", 
                               output_log="output.log", 
                               output_cm='output_cm.h5',
                               report_time=100*u.picoseconds, 
                               sim_time=10000*u.nanoseconds)
    

@app.task
def run_cvae_with_celery(job_id, gpu_id, cvae_input, hyper_dim=3): 
    work_dir = os.getcwd()
    model_dir = os.path.join(work_dir, "cvae_model%d" % int(run_id))
    
    try:
        os.mkdir(model_dir)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass
    
    os.chdir(model_dir) 
    
    cvae = run_cvae(gpu_id, cvae_input, hyper_dim=hyper_dim)
    