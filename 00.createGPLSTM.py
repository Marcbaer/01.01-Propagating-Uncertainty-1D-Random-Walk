'''This script creates the GPLSTM model used by RW_propagation_GPLSTM.py'''

import numpy as np
# Keras
from keras.optimizers import Adam
# Model assembling and executing
from kgp.utils.assemble import load_NN_configs, load_GP_configs, assemble
# Metrics & losses
from kgp.losses import gen_gp_loss

np.random.seed(42)
shift=1

def RandomWalk(N=100, d=1):

    return np.cumsum(np.random.normal(0,0.3162,(N,d)))


def Generate_data(shift):
    """Generate input data of shape (N,d) where d is the sequence length.
    and output data of shape (N,1) for one step ahead predictions.
    """
    
    sample_size=1000
    #1D input data:
    sequence_length=2
    
    total_length=sequence_length+shift
    
    data=RandomWalk(N=sample_size+shift+1,d=1)
 
    #create sequences with length sequence_length
    result = []
    for index in range(len(data) - total_length):
        
        i=data[index: index + total_length]
        k=i[:sequence_length]
        j=np.array(i[total_length-1])
        j=j.reshape(1,)
        k=np.append(k,j,axis=0)
        result.append(k)
        
    result = np.array(result) 

    #shuffle
    np.random.shuffle(result)
  
    #reshape (#Timesteps,seq_length,#modes)
    
    result=result.reshape(result.shape[0],result.shape[1],1)
       
    #sample_size
    Input_data=result[:sample_size,:sequence_length,:]
    Output_data=result[:sample_size,-1,:]
    
    #Train Test valid data split
    train_end = int((50. / 100.) * len(Input_data))
    test_end = int((75. / 100.) * len(Input_data))
    
    X_train=Input_data[:train_end,:,:]
    y_train=Output_data[:train_end,:]
    
    X_test=Input_data[train_end:test_end,:]
    y_test=Output_data[train_end:test_end,:]
    
    X_valid=Input_data[test_end:,:,:]
    y_valid=Output_data[test_end:,:] 
    
    
    data = {
        'train': [X_train, y_train],
        'valid': [X_valid, y_valid],
        'test': [X_test, y_test],
    }
    
    # Re-format targets
    for set_name in data:
        y = data[set_name][1]
        y = y.reshape((-1, 1, np.prod(y.shape[1:])))
        data[set_name][1] = [y[:,:,i] for i in range(y.shape[2])]
    
    return data    
    

def GPLSTM(shift,lr):
    
    data=Generate_data(shift)
    # Model & training parameters
    nb_train_samples = data['train'][0].shape[0]
    input_shape = data['train'][0].shape[1:]
    nb_outputs = len(data['train'][1])
    gp_input_shape = (1,)
    batch_size = 500

    nn_params = {
        'H_dim': 4,
        'H_activation': 'tanh',
        'dropout': 0.0,
    }

    gp_params = {
        'cov': 'SEiso', 
        'hyp_lik': np.log(0.1),
        'hyp_cov': [[0.1], [0.0]],     
    }
    
    # Retrieve model config
    nn_configs = load_NN_configs(filename='lstm.yaml',
                                 input_shape=input_shape,
                                 output_shape=gp_input_shape,
                                 params=nn_params)
    gp_configs = load_GP_configs(filename='gp.yaml',
                                 nb_outputs=nb_outputs,
                                 batch_size=batch_size,
                                 nb_train_samples=nb_train_samples,
                                 params=gp_params)

    # Construct & compile the model
    model = assemble('GP-LSTM', [nn_configs['1H'], gp_configs['GP']]) #MSGP
    loss = [gen_gp_loss(gp) for gp in model.output_layers]
    model.compile(optimizer=Adam(1e-3), loss=loss)
  
    return model

 
