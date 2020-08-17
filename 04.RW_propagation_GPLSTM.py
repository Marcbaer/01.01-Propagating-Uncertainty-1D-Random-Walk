'''Propagate one step ahead GPLSTM mean predictions and variance estimates over multiple steps. Pretrained GPLSTM with shift n=1 required. (Run RW_GPLSTM.py with n=1 before)'''''

import numpy as np
from createGPLSTM import GPLSTM,Generate_data
import pickle
import warnings
warnings.simplefilter(action='ignore')

#test number
test=1
shift=1 #nr of steps into the future
seq_len=2
lr=1e-3 #compiling the model
sample_size=1000
batch_size=100

n_steps=7
start_point=10

#load data
data_shape,RW=Generate_data(shift,sample_size)
data = pickle.load(open('./Results/RW_data_test'+str(test)+'.p', 'rb'),encoding='latin1')
X_test=data['X_test']
X_train=data['X_train']
y_train=data['y_train']
y_test=data['y_test']
RW_initial=data['RW_initial']
data={}
data['train']=[X_train,y_train]

#initialize GPLSTM model
model=GPLSTM(shift,lr,sample_size,batch_size,data_shape)

#load best models for every single mode and finetune, specify checkpoint
model.load_weights('./checkpoints/lstm_1.h5')
model.finetune(*data['train'],batch_size=500,gp_n_iter=100,verbose=0)
print('finetuning done')

#get initial point we want to propagate
X=X_test[start_point:start_point+1,:,:]
X1=np.concatenate((X,X),axis=0) #concatenated only to avoid dimension error

#predict mean&var of initial point
mean_1,var_1 = model.predict(X1,return_var=True, X_tr=X_train, Y_tr=y_train,batch_size=500)
mean_1=np.array(mean_1)[0,1,0]
var_1=var_1[0][0]

#initialize dictionaries to track history
K_d={}
MEAN_d={}
VAR_d={}
X_hist_d={}

#append initial point to history
X_hist_d["hist{0}".format(0)]=[np.array(X),0]
MEAN_d["mean{0}".format(0)]=[mean_1,0]
VAR_d["var{0}".format(0)]=[var_1,0]

# number of sampling points per step
n_samples=80
#number of steps propagating into the future
for n in range(1,n_steps+1):   
        #initialize new history index
        K=[]
        MEAN=[]
        VAR=[]
        X_hist=[]
    
        for j in range(1,n_samples+1,1):
            #sample random integer to get index
            if n==1:
                k=0
            else:
                k=np.random.randint(n_samples)
            
            #get mean, var and history of sampled index
            u=MEAN_d["mean{0}".format(n-1)][k]
            v=VAR_d["var{0}".format(n-1)][k]
            std=v**0.5
            X_old=X_hist_d["hist{0}".format(n-1)][k]
                 
            #sample new point
            k2=np.random.normal(u,std)
            k2=np.array(k2).reshape(1,)
            K.append(k2)
            
            #create history of prediction mode using new sampled point and old history
            X_new1=np.concatenate((X_old[0,1:,0],k2),axis=0)
            X_new1=X_new1.reshape(1,seq_len,1)
         
            #Use new history to predict new mean and var
            X_=np.concatenate((X_old,X_new1),axis=0)    
            mean_2,var_2=model.predict(X_,return_var=True)
            mean_2=np.array(mean_2)[0,1,0]
            var_2=np.array(var_2)[0,1,0]
            
            #append new history to list and predicted mean&var of sampled point
            X_hist.append(X_new1)
            MEAN.append(mean_2)
            VAR.append(var_2)
            print('step {}/{} sample done {}/{}'.format(n,n_steps,j,n_samples))
            
        #append new history to dictionnaries
        K_d["K{0}".format(n)]=np.array(K)
        MEAN_d["mean{0}".format(n)]=np.array(MEAN)
        VAR_d["var{0}".format(n)]=np.array(VAR)
        X_hist_d["hist{0}".format(n)]=np.array(X_hist)    
        print('step done:', n)

#define X-Axis points        
W={}       
for i in range(1,n_steps+1):
    
        W['w{0}'.format(i)]=np.full((n_samples),i)
        
w=np.full((),0)  
X=np.array(X_hist_d['hist0'][0])
X_initial=X[:,-1,0]

#save results
res={'K_d': K_d,'MEAN_d':MEAN_d,'VAR_d':VAR_d,'X_hist_d':X_hist_d,'W':W,'w':w,'mean_1':mean_1,'var_1':var_1,'n_samples':n_samples,'n_steps':n_steps,'RW_initial':RW_initial,'y_test':y_test}
pickle.dump(res, open('./Results/res_propagation_test'+str(test)+'.p', "wb"))