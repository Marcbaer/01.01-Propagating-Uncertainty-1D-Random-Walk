'''Visualization of one dimensional random walks'''

import numpy as np
import matplotlib.pyplot as plt

#np.random.seed(42)

def RandomWalk(N=100, d=1):
    '''
    Use numpy.cumsum and numpy.random.uniform to generate
    a 1D random walk of length N. Each step is sampled from normal distribution
    with mean 0 and Variance 0.1.
    
    Parameters
    ----------
    N : Integer, optional
        Number of steps to be sampled. The default is 100.
    d : Integer, optional
        Number of Dimensions of the Random Walk. The default is 1.

    Returns
    -------
    Numpy Array
    Sampled Random Walk.

    '''
    return np.cumsum(np.random.normal(0,0.3162,(N,d)))

#1D input data:
sample_size=10000
data=RandomWalk(N=sample_size,d=1)

#plot
plt.figure(figsize=(8,6))
plt.title(' 3 independent random walks with %1.f steps' %sample_size)

for i in range(3):
    data=RandomWalk(N=sample_size,d=1)
    data=np.array(data)
    k=np.array(0)
    k=k.reshape(1,)
    data=np.concatenate((k,data))
    plt.plot(data,label='Walk '+str(i+1))
    
plt.legend()
plt.xlabel('#steps')
plt.ylabel('Cumulative value')
plt.savefig('./Figures/randomwalk.pdf')
plt.show()           