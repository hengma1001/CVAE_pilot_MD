from celery.bin import worker
import threading

def start_worker(celery_app): 
    """
    A function starting the celery works within the python script and sending
    the worker running at the background. 
    
    Parameters: 
    -----------
    app : the "Celery" from the project file, mostly defined as 
        app = Celery('tasks', broker='pyamqp://guest@localhost//', backend='rpc://') 
    
    """
    
    celery_worker = worker.worker(app=celery_app)
    threaded_celery_worker = threading.Thread(target=celery_worker.run) # , args=('--concurrency=4', ))
    threaded_celery_worker.daemon = 1
    threaded_celery_worker.start()