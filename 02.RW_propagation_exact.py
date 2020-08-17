'''Evaluation of the propagation algorithm: Propagate theoretic exact values'''

import numpy as np
import matplotlib.pyplot as plt

#get initial point we want to propagate
X=0
#predict mean&var of initial point
mean_1=np.random.normal(X,0.3162)

#initialize dictionaries to track history
K_d={}
MEAN_d={}
X_hist_d={}

#append initial point to history
X_hist_d["hist{0}".format(0)]=[np.array(X)]
MEAN_d["mean{0}".format(0)]=[mean_1]

# number of sampling points per step
n_samples=1000
#number of steps propagating into the future
n_steps=10

for n in range(1,n_steps+1):   
        #initialize new history index
        K=[]
        MEAN=[]
        
        for j in range(1,n_samples+1,1):
            #sample random integer to get index
            if n==1:
                k=0
            else:
                k=np.random.randint(n_samples)
            
            #get mean, var and history of sampled index
            u=MEAN_d["mean{0}".format(n-1)][k]
                             
            #sample new point
            k2=np.random.normal(u,0.3162)
            k2=np.array(k2).reshape(1,)
            K.append(k2)
            
            #create history of prediction mode using new sampled point and old history
            #Use new history to predict new mean and var
            mean_2=k2
            
            #append new history to list and predicted mean&var of sampled point
            MEAN.append(mean_2)
               
        #append new history to dictionnaries
        K_d["K{0}".format(n)]=np.array(K)
        MEAN_d["mean{0}".format(n)]=np.array(MEAN)

        #X_hist_d["hist{0}".format(n)]=np.array(X_hist)    
        print('step done:', n)

#plotting
#define X-Axis points        
W={}       
for i in range(1,n_steps+1):
    
        W['w{0}'.format(i)]=np.full((n_samples),i)
        
w=np.full((),0)  

#plot initial starting point with errorbar

X_initial=0
plt.scatter(w,X_initial,color='blue',label='distribution of sampled points')

#boundaries
bound1=[0]
bound2=[0]
var1=[]
x=[0,1,2,3,4,5,6,7,8,9,10.3]
for i in range(1,n_steps+1):
    
    initial_var=0.1
    var=i*initial_var
    var1.append(var)
    std=var**0.5
    bound=3.5*std
    bound1.append(X_initial+bound)
    bound2.append(X_initial-bound)
    
plt.plot(x,bound1,color='red',label='Theoretic 95% confidence')
plt.plot(x,bound2,color='red')
plt.fill_between(x,bound1,bound2,color='lightgrey')


for i in range(1,n_steps+1):
    pos=[i]
    violin=plt.violinplot(K_d['K{0}'.format(i)].reshape(len(K_d['K{0}'.format(i)])),pos,showmeans = True)
    #print('variance of step:',i,' is:', K_d['K{0}'.format(i)].reshape(len(K_d['K{0}'.format(i)])).var())
    for pc in [violin['cbars'],violin['cmins'],violin['cmaxes'],violin['cmeans']]:
        pc.set_edgecolor('#2222ff')
    for pc in violin['bodies']:
        pc.set_facecolor('#2222ff')
                      
plt.grid(False)
plt.xlabel('#step')
plt.title('Random walk propagation with theoretic 95 % confidence')
plt.legend(loc=2,prop={'size': 6})
plt.savefig('./Figures/RWpropexact.pdf')
