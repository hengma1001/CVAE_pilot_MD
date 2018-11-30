from CVAE import CVAE

import gc, time, GPUtil

import keras 

model = CVAE((22,22,1), 4) 

model.model.summary()

print 'created the CVAE model.' 

GPUtil.showUtilization()


print 'trying to clear the GPU memory now.' 

keras.backend.clear_session() 
print 'tried K.clear_session, check memory in 30 s.' 
GPUtil.showUtilization() 

del model.model
print 'tried delete model, check memory in 30 s.'
GPUtil.showUtilization()

gc.collect()
print 'tried gc, check memory in 30 s.'
GPUtil.showUtilization()


time.sleep(30)
GPUtil.showUtilization()

model_new = CVAE((22,22,1), 4)

time.sleep(1000)
