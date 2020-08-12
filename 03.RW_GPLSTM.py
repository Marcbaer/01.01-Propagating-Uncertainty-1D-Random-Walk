'''This script trains a GPLSTM and predicts with shift n. Including prediction plots, variance and step-size histogramms '''

import numpy as np
from createGPLSTM import GPLSTM,Generate_data

# Model assembling and executing
from kgp.utils.experiment import train
# Metrics & losses
from kgp.metrics import root_mean_squared_error as RMSE
import pickle
import matplotlib.pyplot as plt
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
    
#np.random.seed(5)

#test number
test=1
shift=1 #nr of steps into the future
seq_len=2
lr=1e-3 #compiling the model

epochs=20
sample_size=1000
batch_size=100

def main(shift,sample_size,batch_size,epochs):
    '''Create GPLSTM Model and Train it on the Random Walk
       Returns: Model and Training Results'''
    
    data=Generate_data(shift,sample_size)
    
    # Model & training parameters
    model=GPLSTM(shift,lr,sample_size,batch_size)
    
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

    history,X_test,X_train,y_train,batch_size,y_test,model=main(shift,sample_size,batch_size,epochs)

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















