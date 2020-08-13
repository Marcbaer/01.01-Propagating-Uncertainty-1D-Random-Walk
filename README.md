# 01.Propagating-Uncertainty-in-Dynamical-Systems

This experiment evaluates GP-LSTM's (Gaussian Process regression in combination with LSTM's) on their ability to forecast the predictive distribution of dynamical systems.
The GP-LSTM models are built using the keras-gp library (https://github.com/alshedivat/keras-gp) with an octave engine.

*Evaluation of predicted variances*
The evaluation of the predicted distribution is based on square root of time rule.
According to the square root of time rule for simple random walks, the variance for a n-th step ahead prediction <img src="https://render.githubusercontent.com/render/math?math=\sigma_{n}^{2}">
is the sum of the variances of each
individual taken step and hence <img src="https://render.githubusercontent.com/render/math?math=n*\sigma_{1}^{2}">. 
In other words, variances are added if independent and identically distributed random variables are added.

*01.01 One Dimensional Random Walk*

The variance estimates of the GP-LSTM model are evaluated on a simple one
dimensional random walk. The data is generated by sampling a random step
from a normal distribution with mean 0 and variance <img src="https://render.githubusercontent.com/render/math?math=\sigma^{2}">.

If <img src="https://render.githubusercontent.com/render/math?math=S_{t}"> represents the value of the random walk at time *t*, the sequence is described by:

<img src="https://render.githubusercontent.com/render/math?math=S_{t+1} \ = \ S_{t} $+$ \epsilon , \epsilon \sim \mathcal{N}(0,\,\sigma_{1}^{2})">.

The GP-LSTM is trained for one-step-ahead predictions using a random walk with 1000 time steps.
An uncertainty propagation algorithm is applied to propagate predictive uncertainties n-steps into the future by sampling a n_samples per time step using the predicted distribution.