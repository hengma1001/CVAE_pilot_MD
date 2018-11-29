from CVAE import CVAE

import gc, time, GPUtil

import keras 
import tensorflow as tf

model = CVAE((22,22,1), 4) 

model.model.summary()

print 'created the CVAE model.' 

GPUtil.showUtilization()


print 'trying to clear the GPU memory now.' 

del model.model
del model
print 'tried delete model, check memory in 30 s.'
GPUtil.showUtilization()

tf.reset_default_graph()
keras.backend.clear_session() 
time.sleep(5)
print 'tried K.clear_session, check memory in 30 s.' 
GPUtil.showUtilization() 


gc.collect()
print 'tried gc, check memory in 30 s.'
time.sleep(5)
GPUtil.showUtilization()


time.sleep(5)
GPUtil.showUtilization()

model_new = CVAE((22,22,1), 4)

time.sleep(1000)
