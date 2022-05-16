# use block-coordinate descent (BCD)
# optimize group of vars while fixing other groups
# n+2 blocks: X,Y,M1,M2,...Mn
import numpy as np
import pdb
import time

def compute_x(Xsum, alphas):
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
  return D,n

def compute_M(Mopt,gamma,alpha,beta,X,Y):
  tau = gamma/(alpha + beta)
  Z = (alpha * X + beta * Y)/ (alpha + beta)
  # reshape Z from H,W,C -> H,W*C
  Z = np.reshape(Z,(Z.shape[0],Z.shape[1]*Z.shape[2]))
  Mopt,di = shrinkage(Z, tau)
  Mopt = np.reshape(Mopt,X.shape)
  return Mopt

#assume T is of shape HxWxC (channel last)
def lrtc(T,alpha,beta,gamma,max_itr=10):
  Y= np.copy(T)
  mask = (T > 250) *1
  X = np.copy(Y)
  M = []
  for i in range(T.shape[-1]):
    M.append(np.copy(Y))

  err = []
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
    Ysum = compute_y(Ysum,beta)
    #pdb.set_trace()
    print("Y - Y_bar= " + str((Y-Ysum).sum()))
    err.append([(Y-Ysum).sum()])
    Y[mask==1] = Ysum[mask==1]
    itr += 1
    if itr == max_itr:
      convergence = True
    #TODO-- would be more elegant if we had some threshold value
  end_time = time.perf_counter()
  print("Time taken for LRTC: " + str(end_time-start_time) + "s")
  return Y,err
