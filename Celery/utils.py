from tasks import run_omm_with_celery, run_cvae_with_celery
from celery.bin import worker
import numpy as np
import threading, h5py
import subprocess
from molecules.utils.matrix_op import triu_to_full

def read_h5py_file(h5_file): 
    cm_h5 = h5py.File(h5_file, 'r', libver='latest', swmr=True)
    return cm_h5[u'contact_maps'] 

def start_rabbit(rabbitmq_log): 
    """
    A function starting the rabbitmq server within the python script and sending
    the worker running at the background. 
    
    Parameters: 
    -----------
    rabbitmq_log : ``str``
        log file contains the screen output of rabbitmq server 
    
    """
    log = open(rabbitmq_log, 'w')
    subprocess.Popen('rabbitmq-server', stdout=log, stderr=log) 

def start_worker(celery_worker_log): 
    """
    A function starting the celery works within the python script and sending
    the worker running at the background. 
    
    Parameters: 
    -----------
    celery_worker_log : ``str``
        log file contains the screen output of celery worker 
    
    """
    
    celery_cmdline = "celery worker -A tasks" 
    log = open(celery_worker_log, 'w')
    subprocess.Popen(celery_cmdline.split(" "), stdout=log, stderr=log) 
    # This format of starting the workers used mess up the print function in notebook. 
#     celery_worker = worker.worker(app=celery_app)
#     threaded_celery_worker = threading.Thread(target=celery_worker.run) 
#     threaded_celery_worker.start() 
#     return threaded_celery_worker
    

def start_flower_monitor(address='127.0.0.1', port=5555): 
    """
    A function starting the flower moniter for celery servers and workers. 
    The information is available at http://127.0.0.1:5555 by default
    
    Parameters: 
    -----------
    address : ``string``
        The address to the flower server
    port : ``int``
        The port to open or port the server 
        
    """ 
    
    celery_flower_cmdline = 'celery flower -A tasks --address={0} --port={1}'.format(address, port)
    subprocess.Popen(celery_flower_cmdline.split(" "))
    

def cm_to_cvae(cm_data_lists): 
    """
    A function converting the 2d upper triangle information of contact maps 
    read from hdf5 file to full contact map and reshape to the format ready 
    for cvae
    """
    cm_all = np.hstack(cm_data_lists)

    # transfer upper triangle to full matrix 
    cm_data_full = np.array([triu_to_full(cm_data) for cm_data in cm_all.T]) 

    # padding if odd dimension occurs in image 
    pad_f = lambda x: (0,0) if x%2 == 0 else (0,1) 
    padding_buffer = [(0,0)] 
    for x in cm_data_full.shape[1:]: 
        padding_buffer.append(pad_f(x))
    cm_data_full = np.pad(cm_data_full, padding_buffer, mode='constant')

    # reshape matrix to 4d tensor 
    cvae_input = cm_data_full.reshape(cm_data_full.shape + (1,))   
    
    return cvae_input


def job_on_gpu(gpu_id, jobs): 
    """
    Find job on GPU gpu_id
    
    Parameters: 
    -----------
    gpu_id : ``int`` 
    jobs : ``list of celery tasks``
    """
    for job in jobs: 
        if job.gpu_id == gpu_id: 
            return job 
        

class omm_job(object): 
    """
    A OpenMM simulation job. 
    
    Parameters: 
    -----------
    job_id : ``int`` 
        A int number to track the job, according to which the job will create a directory 
        and store the log, trajectory and contact maps h5 files 
    gpu_id : ``int``
        The id of GPU, on which the OpenMM will be running 
    top_file : ``str``
        The location of input topology file for OpenMM 
    pdb_file : ``str``
        The location of input coordinate file for OpenMM 
        
    """
    def __init__(self, job_id=0, gpu_id=0, top_file=None, pdb_file=None, check_point=None): 
        self.job_id = job_id
        self.gpu_id = gpu_id
        self.top_file = top_file
        self.pdb_file = pdb_file 
        self.check_point = None
        self.job = None 
        
    def start(self): 
        """
        A function to start the job and store the `class :: celery.result.AsyncResult` 
        in the omm_job.job 
        """
        sim_job = run_omm_with_celery.delay(self.job_id, self.gpu_id, 
                                       self.top_file, self.pdb_file, 
                                       self.check_point) 
        self.job = sim_job
    
    def stop(self): 
        """
        A function to stop the job and return the available gpu_id 
        """
        if self.job: 
            self.job.revoke(terminate=True) 
        else: 
            raise Exception('Attempt to stop a job, which is not running. \n')
        return self.gpu_id 
    

    
class cvae_job(object): 
    """
    A CVAE job. 
    
    Parameters: 
    -----------
    job_id : ``int`` 
        A int number to track the job, according to which the job will create a directory 
        and store the weight files 
    gpu_id : ``int``
        The id of GPU, on which the CVAE will be running 
    input_data_file : ``str`` file location
        The location of h5 file for CVAE input  
    hyper_dim : ``int``
        The number of latent space dimension 
        
    """
    def __init__(self, job_id, gpu_id=0, cvae_input=None, hyper_dim=3): 
        self.job_id = job_id
        self.gpu_id = gpu_id
        self.cvae_input = cvae_input
        self.hyper_dim = hyper_dim
        self.job = None 
        
    def start(self): 
        """
        A function to start the job and store the `class :: celery.result.AsyncResult` 
        in the cvae_job.job 
        """
        sim_job = run_cvae_with_celery.delay(self.job_id, self.gpu_id, 
                                             self.cvae_input, hyper_dim=self.hyper_dim)
        self.job = sim_job 
        
    def cave_model(self): 
        pass
#         if self.job.
    
    def _stop(self): 
        """
        A function to stop the job and return the available gpu_id 
        """
        if self.job: 
            self.job.revoke(terminate=True) 
        else: 
            raise Exception('Attempt to stop a job, which is not running. \n')
