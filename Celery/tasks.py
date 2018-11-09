from celery import Celery
import simtk.unit as u
import sys, os
import errno

sys.path.append('/home/hm0/Research/molecules/molecules_git/build/lib')
from molecules.sim.openmm_simulation import openmm_simulate_charmm_nvt

app = Celery('tasks', broker='pyamqp://guest@localhost//', backend='rpc://') 



@app.task
def add(x, y):
    return x + y

@app.task
def run_omm_on_gpu(run_id, gpu_index, top_file, pdb_file): 
    work_dir = os.getcwd()
    iter_dir = os.path.join(work_dir, "omm_run%d" % int(run_id))
    
    try:
        os.mkdir(iter_dir)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass
    
    os.chdir(iter_dir) 

    openmm_simulate_charmm_nvt(top_file, pdb_file, GPU_index=gpu_index, 
                               output_traj="output.dcd", 
                               output_log="output.log", 
                               output_cm='output_cm.h5',
                               report_time=100*u.picoseconds, 
                               sim_time=1000*u.nanoseconds)
    

@app.task
def cvae(): 
    pass