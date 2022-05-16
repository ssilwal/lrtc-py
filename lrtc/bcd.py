# use block-coordinate descent (BCD)
# optimize group of vars while fixing other groups
# n+2 blocks: X,Y,M1,M2,...Mn
import numpy as np
import pdb
import time

def compute_x(Xsum, alphas):
  # a - alpha weights (there should be n values)
  # M - array of Mi values (there should be n)
  
  #X = Sigma (a_i fold_i(M_i)) / Sigma (a_i) 
  X = Xsum/ (alphas.sum() + 1e-4)
  return X


def compute_y(Ysum,betas):
  Y = Ysum/ (betas.sum() + 1e-4)
  return Y


def shrinkage(X,tau):
  #SVD X
  U,Sigma,V = np.linalg.svd(X) 
  n = Sigma.shape[0]
  
  # ||X||_tr = Sum(sigma_i(X))
  sigma = np.linalg.svd(X)[1]

  Sigma_tau = np.diag(np.maximum(sigma-tau, np.zeros((n,n))))
  n = (Sigma_tau>0).sum()
  #D =  np.dot(U,np.dot(Sigma_tau,V[:n,:n]))
  #D = U[:,:n] * Sigma_tau * V[:n,:n]
  D = np.dot(U*Sigma_tau, V[:Sigma_tau.shape[0],:])
  return D

def compute_M(Mopt,gamma,alpha,beta,X,Y):
  tau = gamma/(alpha + beta)
  Z = alpha * X + beta * Y/ (alpha + beta)
  # reshape Z from H,W,C -> H,W*C
  Z = np.reshape(Z,(Z.shape[0],Z.shape[1]*Z.shape[2]))
  Mopt = shrinkage(Z, tau)
  Mopt = np.reshape(Mopt,X.shape)
  return Mopt

#assume T is of shape HxWxC (channel last)
def lrtc(T,gamma,max_itr=10):
  Y= T
  X = Y
  M = []
  for i in range(T.shape[-1]):
    M.append(np.array(Y))

  alpha = np.ones((np.shape(X)[-1])) 
  #alpha[1] = 0.01
  #alpha[2] = 0.001

  beta = np.ones((np.shape(Y)[-1]))
  #beta[0] = 0.1
  #beta[1] = 0.01
  #beta[2] = 0.00001
  convergence = False
  itr = 0

  #measures in seconds https://realpython.com/python-timer/
  start_time = time.perf_counter()
  while not convergence:
    Xsum = np.zeros(X.shape)
    Ysum = np.zeros(Y.shape)
    for i in range(len(M)):
      M[i] = compute_M(M[i],gamma[i],alpha[i],beta[i],X,Y)
      Xsum += alpha[i] *M[i]
      Ysum += beta[i] *M[i]
    X = compute_x(Xsum,alpha)
    Y = compute_y(Ysum,beta)
    print("T - Y_bar= " + str((T-Y).sum()))
    itr += 1
    if itr == max_itr:
      convergence = True
  end_time = time.perf_counter()
  print("Time taken for LRTC: " + str(end_time-start_time) + "s")
  return Y 
