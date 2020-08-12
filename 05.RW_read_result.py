'''Reads the resulting output file from 04.RW_propagation_GPLSTM.py '''

import numpy as np
import matplotlib.pyplot as plt
import pickle
#plotting parameter
plt.rcParams["figure.figsize"] = [4.0, 3.0]

#load results
test=1
start_point=10
steps=7

end_point=start_point+steps

results = pickle.load(open('./Results/res_propagation_test'+str(test)+'.p', 'rb'))

RW_initial=np.array(results['RW_initial'])
K_d=results['K_d']
MEAN_d=results['MEAN_d']
VAR_d=results['VAR_d']
X_hist_d=results['X_hist_d']
W=results['W']
w=results['w']
mean_1=results['mean_1']
var_1=results['var_1']
y_test=results['y_test']
n_steps=results['n_steps']
n_samples=results['n_samples']

X=np.array(X_hist_d['hist0'][0])
X_initial=X[:,-1,0]

#boundaries
bound1=[X_initial]
bound2=[X_initial]
var1=[]

for i in range(1,n_steps+1):
    
    initial_var=0.1
    var=i*initial_var
    var1.append(var)
    std=var**0.5
    bound=4.4*std
    bound1.append(X_initial+bound)
    bound2.append(X_initial-bound)
    
x=list(range(0,len(bound1),1))
x[-1]=x[-1]+0.3
bound1=np.array(bound1).reshape(len(bound1,))
bound2=np.array(bound2).reshape(len(bound2,))    
plt.plot(x,bound1,color='red',label='Theoretic 99.999% confidence')
plt.plot(x,bound2,color='red')
plt.fill_between(x,bound1,bound2,color='lightgrey')


#plot initial starting point with errorbar

plt.scatter(w,X_initial,color='blue',label='sampled distribution')
std1=var_1**0.5
std1=np.array(std1)
std1=3.5*std1

#plot sampled points

for i in range(1,n_steps+1):
    pos=[i]
    violin=plt.violinplot(K_d['K{0}'.format(i)].reshape(len(K_d['K{0}'.format(i)])),pos,showmeans = True)
    for pc in [violin['cbars'],violin['cmins'],violin['cmaxes'],violin['cmeans']]:
        pc.set_edgecolor('#2222ff')
    for pc in violin['bodies']:
        pc.set_facecolor('#2222ff')
                      
plt.grid(False)
plt.xlabel('#step')
plt.title('Random walk propagation')
plt.legend(loc=2,prop={'size': 6})
plt.savefig('./Figures/RW_Uncertainty_nsamples_'+str(n_samples)+'_nsteps_'+str(n_steps)+'.png')
plt.show()


'''RW plot'''  

index=int(np.where(RW_initial==X_initial)[0])
RW=RW_initial[index+1:index+1+n_steps]

data=y_test[0,start_point:start_point+steps,:]
x=[]
y1=[X_initial]
y2=[X_initial]
mean1=[]

for i in range(1,n_steps+1):
    pos=[i]
    y1.append(min(K_d['K{0}'.format(i)]))
    y2.append(max(K_d['K{0}'.format(i)]))
    x.append(i)
    mean1.append(K_d['K{0}'.format(i)].mean())
    
x1=np.append(0,x)

y1=np.array(y1).reshape(len(x1),)
y2=np.array(y2).reshape(len(x1),)    
plt.plot(x,mean1,label='predicted mean',color='blue',marker='o',linestyle='--')
plt.plot(x,RW,label='Random Walk',marker='o',linestyle='--',color='red')
plt.fill_between(x1,y1,y2,facecolor='lightgrey',label='confidence bound')

plt.scatter(0,X_initial,color='green',label='initial point')

plt.xlabel('#step')
plt.title('Predicted mean vs. true values')
plt.legend(loc=2,prop={'size': 6})
plt.savefig('./Figures/RW_path_propagation_test_{}.png'.format(test))

