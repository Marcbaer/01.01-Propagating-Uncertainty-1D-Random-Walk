'''This script trains a GPLSTM and predicts with shift n. Including prediction plots, variance and step-size histogramms '''

from __future__ import print_function
import numpy as np
# Keras
from keras.optimizers import Adam
# Model assembling and executing
from kgp.utils.experiment import train
# Metrics & losses
from kgp.losses import gen_gp_loss
from kgp.metrics import root_mean_squared_error as RMSE
import pickle
import matplotlib.pyplot as plt
from kgp.utils.assemble import load_NN_configs, load_GP_configs, assemble
import matplotlib
from matplotlib.ticker import StrMethodFormatter
import warnings
warnings.simplefilter(action='ignore')


#Plotting parameters
plt.rcParams["figure.figsize"] = [4., 3.]
SMALL_SIZE = 10
matplotlib.rc('axes', titlesize=SMALL_SIZE)
matplotlib.rc('font', size=SMALL_SIZE)
plt.tick_params(labelsize=10)
    
np.random.seed(5)

shift=1
test=1

def RandomWalk(N=100, d=1):
    '''Creates 1 dimensional Random Walk of length N
       Returns: Random Walk'''
    return np.cumsum(np.random.normal(0,0.3162,(N,d)))


def Generate_data(shift):
    """Generate input data of shape (N,d) where d is the sequence length.
    and output data of shape (N,1) for one step ahead predictions.
    Returns: Data for training and testing the model
    """
    
    sample_size=1000

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

    #reshape (#Timesteps,seq_length,#modes)
    
    result=result.reshape(result.shape[0],result.shape[1],1)
    
    train_end=int(0.8*len(result))
    res_train=result[:train_end]
    res_test=result[train_end:]
    
    np.random.shuffle(res_train)
    
    #sample_size
    valid=int(0.8*len(res_train))
    Input_data=res_train[:sample_size,:sequence_length,:]
    Output_data=res_train[:sample_size,-1,:]

    Input_data_test=res_test[:sample_size,:sequence_length,:]
    Output_data_test=res_test[:sample_size,-1,:]  
    
    X_train=Input_data[:valid,:,:]
    y_train=Output_data[:valid,:]
    
    X_test=Input_data_test[:,:]
    y_test=Output_data_test[:,:]
    
    X_valid=Input_data[valid:,:,:]
    y_valid=Output_data[valid:,:] 
    
    
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
    

def main(shift):
    '''Create GPLSTM Model and Train it on the Random Walk
       Returns: Model and Training Results'''
    
    data=Generate_data(shift)
    
    # Model & training parameters
    nb_train_samples = data['train'][0].shape[0]
    input_shape = data['train'][0].shape[1:]
    nb_outputs = len(data['train'][1])
    gp_input_shape = (1,)
    batch_size = 50
    epochs = 20

    nn_params = {
        'H_dim': 4,
        'H_activation': 'tanh',
        'dropout': 0.0,
    }

    gp_params = {
        'cov': 'SEiso', 
        'hyp_lik': np.log(0.1),
        'hyp_cov': [[1.0], [1.0]],
        'inf': 'infExact',
        'lik': 'likGauss',
        'dlik': 'dlikExact',             
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
    model = assemble('GP-LSTM', [nn_configs['1H'], gp_configs['GP']])
    loss = [gen_gp_loss(gp) for gp in model.output_layers]
    model.compile(optimizer=Adam(1e-3), loss=loss)

    
    callbacks = []
    
    history = train(model, data, callbacks=callbacks, gp_n_iter=5,
                    checkpoint='lstm_1', checkpoint_monitor='val_mse',
                    epochs=epochs, batch_size=batch_size, verbose=1)
    
    # Finetune the model
    model.finetune(*data['train'],
                   batch_size=batch_size,
                   gp_n_iter=100,
                   verbose=1)
    
    # Test the model
    X_test,y_test =data['test']
    X_train,y_train=data['train']
           
    return history,X_test,X_train,y_train,batch_size,y_test,model


if __name__ == '__main__':

    history,X_test,X_train,y_train,batch_size,y_test,model=main(shift)

    y_pred,var = model.predict(X_test,return_var=True, X_tr=X_train, Y_tr=y_train,batch_size=batch_size)
    
    y_pred=np.array(y_pred)
    y_test=np.array(y_test)
    var=np.array(var)
    std=[i**0.5 for i in var]
    std=np.array(std)
    std2=2*std
 
    rmse_predict = RMSE(y_test, y_pred)
    
    validation_error=history.history['val_mse']
    training_error=history.history['mse']
    training_error=np.array(training_error)
    training_error=training_error/y_test.shape[0]
    
    res={'X_test':X_test,'X_train':X_train,'y_train':y_train,'y_test':y_test} 
    pickle.dump(res, open('./Results/RW_Data_test'+str(test)+'.p', "wb"))
    
    #Training convergence
    plt.figure(figsize=(8,6))
    plt.xlabel("# Epoch")
    plt.ylabel('RMSE')
    plt.title('MSGP training convergence')
    plt.plot(validation_error,label='validation_error')
    plt.plot(training_error,label='training_error')
    plt.legend()
    plt.savefig('./Figures/Training_convergence')
    plt.show()

    #one step ahead
    #var=1 std, to get 95% confidence (2 std in each direction), so var*2=95% confidence
    SMALL_SIZE = 18
    matplotlib.rc('axes', titlesize=SMALL_SIZE)
    matplotlib.rc('font', size=SMALL_SIZE)
    
    plt.tick_params(labelsize=25)
    start=10
    size=20
    J=np.arange(0,y_test[0,start:start+size].shape[0],1)
    plt.figure(figsize=(8,6))
    plt.title('Predictive distribution for 1 step ahead predictions')
    plt.xlabel("Test point")
    plt.errorbar(J,y_pred[0,start:start+size,0],yerr=std2[0,start:start+size,0],capsize=0,fmt='',ecolor='grey',label='Predicted mean and 95% confidence',marker='.',markersize=10,ls='none')
    plt.scatter(J,y_test[0,start:start+size],label='True value',color='red',marker='o')
    plt.legend(loc=2, prop={'size': 12})
    plt.savefig('./Figures/RW_pred_shift1.pdf')
    plt.show()
    
    print('max variance: ',max(var[0,:]),' min variance: ', min(var[0,:]))
    print('RMSE: ', rmse_predict)

    #plot estimated step-size, should be distributed around 0
    
    k=abs(X_test[:,-1,:])-abs(y_pred)
    k=k[0,:,0]
    i=len(k)
    
    plt.figure(figsize=(8,6))
    plt.title('Step-size histogram: 1 step ahead prediction')
    plt.hist(k,bins=30)
    plt.xlabel('step size')
    plt.ylabel('#points')
    plt.savefig('./Figures/Grid_step_hist_1.pdf')
    plt.show()
    
    #plot estimated variances, should be distributed around 0.1
    
    var=var[0,:,0]
    
    plt.figure(figsize=(8,6))
    plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:,.4f}'))
    plt.title('Variance estimates: 1 step ahead prediction')
    plt.hist(var,bins=30,label='min_variance='+str(round(min(var),3)))
    plt.xlabel('variance')
    plt.ylabel('#points')
    plt.legend()
    plt.savefig('./Figures/Grid_var_hist_1.pdf')
    plt.show()    















