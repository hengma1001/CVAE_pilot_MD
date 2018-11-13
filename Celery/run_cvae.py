import os, sys, h5py
# from utils import CVAE 
from keras.optimizers import RMSprop

sys.path.append('/home/hm0/Research/molecules/molecules_git/build/lib')
from molecules.ml.unsupervised import VAE
from molecules.ml.unsupervised import EncoderConvolution2D
from molecules.ml.unsupervised import DecoderConvolution2D
from molecules.ml.unsupervised.callbacks import EmbeddingCallback 

def CVAE(input_shape, hyper_dim=3): 
    optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

    encoder = EncoderConvolution2D(input_shape=input_shape)

    encoder._get_final_conv_params()
    num_conv_params = encoder.total_conv_params
    encode_conv_shape = encoder.final_conv_shape

    decoder = DecoderConvolution2D(output_shape=input_shape,
                                   enc_conv_params=num_conv_params,
                                   enc_conv_shape=encode_conv_shape)

    cvae = VAE(input_shape=input_shape,
               latent_dim=hyper_dim,
               encoder=encoder,
               decoder=decoder,
               optimizer=optimizer) 
    return cvae 

def run_cvae(gpu_id, cm_file, hyper_dim=3): 
    # read contact map from h5 file 
    cm_h5 = h5py.File(cm_file, 'r', libver='latest', swmr=True)
    cm_data_input = cm_h5[u'contact_maps'] 

    # splitting data into train and validation
    train_val_split = int(0.8 * len(cm_data_input))
    cm_data_train, cm_data_val = cm_data_input[:train_val_split], cm_data_input[train_val_split:] 
    input_shape = cm_data_train.shape[1:] 
    cm_h5.close()

    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu_id) 
    
    cvae = CVAE(input_shape, hyper_dim)
    
    callback = EmbeddingCallback(cm_data_train, cvae)
    cvae.train(cm_data_train, validation_data=cm_data_val, batch_size=512, epochs=100, callbacks=[callback]) 
    
    return cvae 