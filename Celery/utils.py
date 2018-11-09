from celery.bin import worker
import threading
import subprocess

def start_worker(celery_app): 
    """
    A function starting the celery works within the python script and sending
    the worker running at the background. 
    
    Parameters: 
    -----------
    app : ``Celery``
        the `Celery` from the project file, mostly defined as 
        app = Celery('tasks', broker='pyamqp://guest@localhost//', backend='rpc://') 
    
    Examples: 
    ---------
    """
    
    celery_worker = worker.worker(app=celery_app)
    threaded_celery_worker = threading.Thread(target=celery_worker.run) # , args=('--concurrency=4', ))
    threaded_celery_worker.daemon = 1
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