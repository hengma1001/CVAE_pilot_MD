from tasks import run_omm_with_celery
from celery.bin import worker
import threading, h5py
import subprocess


def read_h5py_file(h5_file): 
    cm_h5 = h5py.File(h5_file, 'r', libver='latest', swmr=True)
    return cm_h5[u'contact_maps'] 


def start_worker(celery_app): 
    """
    A function starting the celery works within the python script and sending
    the worker running at the background. 
    
    Parameters: 
    -----------
    app : ``Celery``
        the `Celery` from the project file, mostly defined as 
        app = Celery('tasks', broker='pyamqp://guest@localhost//', backend='rpc://') 
    
    """
#     celery_cmdline = "celery worker -A tasks" 
#     subprocess.Popen(celery_cmdline.split(" ")) 
    # This format of starting the workers used mess up the print function in notebook. 
    celery_worker = worker.worker(app=celery_app)
    threaded_celery_worker = threading.Thread(target=celery_worker.run) 
    threaded_celery_worker.start() 
    return threaded_celery_worker
    

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
        sim_job = run_omm_on_gpu.delay(self.job_id, self.gpu_id, 
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
        and store the log, trajectory and contact maps h5 files 
    gpu_id : ``int``
        The id of GPU, on which the OpenMM will be running 
    top_file : ``str``
        The location of input topology file for OpenMM 
    pdb_file : ``str``
        The location of input coordinate file for OpenMM 
        
    """
    def __init__(self, job_id, gpu_id=0, training_date=None): 
        self.job_id = job_id
        self.gpu_id = gpu_id
        self.training_data = traning_data
        self.job = None 
        
    def start(self): 
        """
        A function to start the job and store the `class :: celery.result.AsyncResult` 
        in the omm_job.job 
        """
#         sim_job = run_omm_on_gpu.delay(self.job_id, self.gpu_id, 
#                                         self.top_file, self.pdb_file) 
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