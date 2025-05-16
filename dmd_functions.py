#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 19:49:45 2024

@author: anandita
"""

import numpy as np
import scipy


def delay_coordinates(x_t,L):
    '''
    

    Parameters
    ----------
    x_t : Time series of length T+1. Can be a one d timeseries or a vector time
          series
    L : No. of time-delays for delay embedding.
    Returns
    -------
    x_delay: L times T matrix starting from x0 in case of 1d timeseries or
               nL times T matrix when x is in R^n
    
    '''
    if np.ndim(x_t)==1:
        T=len(x_t)
        x_delay=np.vstack([x_t[i:T-L+i] for i in range(L+1)])
        
    elif np.ndim(x_t)==2:
        T=len(x_t[0])
        x_delay=np.vstack([x_t[:,i:T-L+i] for i in range(L+1)])
        
    return x_delay


    

def dynamic_matrix_delay_coods(x_t,L):
    assert np.ndim(x_t)==2
    T=len(x_t[0])
    n=len(x_t)
    x0_delay,x1_delay=delay_coordinates(x_t, L)
    A=np.zeros((n*L,n*L))
    for i in range(0,n*L,n):
        A[i:i+n,0:i+n]=np.matmul(x1_delay[i:i+n],np.linalg.pinv(x0_delay[0:i+n]))
    return A

def eDMD(x_t,L):
    assert np.ndim(x_t)==2
    T=len(x_t[0])
    n=len(x_t)
    x0_delay,x1_delay=delay_coordinates(x_t, L)
    A=np.matmul(x1_delay,np.linalg.pinv(x0_delay))
    return A

def proj_A_eDMD(A,n,L):
    return A[((L-1)*n):]

def discrete_RNN(x0,T,n,phi,J):
    x_t=np.zeros((n,T))
    x_t[:,0]=x0
    
    for i in range(1,T):
        x_t[:,i]=np.matmul(J,phi(x_t[:,i-1]))
    return x_t


def dmd_with_control_with_var_cutoff(X,Y,var_cutoff):
    X0=X[:,:-1]
    X1=X[:,1:]
    
    n=X0.shape[0]
    assert X0.shape==X1.shape and X0.shape[1]==Y.shape[1]
    
    Omega=np.vstack((X0,Y))
    
    U,Sigma,V=np.linalg.svd(Omega,full_matrices=False)
    
    exp_var_ratio=np.cumsum(Sigma**2)/np.sum(Sigma**2)
    p=np.searchsorted(exp_var_ratio, var_cutoff)+1
    
    U_bar1,Sigma_bar,V_bar=U[:n,:p],Sigma[:p],V[:p]
    
    U_bar2=U[n:,:p]
    
    A_bar=np.linalg.multi_dot((X1,V_bar.T,np.diag(1/Sigma_bar),U_bar1.T))
    B_bar=np.linalg.multi_dot((X1,V_bar.T,np.diag(1/Sigma_bar),U_bar2.T))
    
    
    eigenvals_X1,U_tilde=np.linalg.eig(X1@X1.T)
    
    exp_var_ratio_x1=np.cumsum(eigenvals_X1)/np.sum(eigenvals_X1)
    r=np.searchsorted(exp_var_ratio_x1, var_cutoff)+1
    
    
    U_tilde=U_tilde[:,:r]
    
    A_red=np.linalg.multi_dot((U_tilde.T,A_bar,U_tilde))
    
    B_red=np.matmul(U_tilde.T,B_bar)
    
    return A_bar,B_bar,U_tilde, A_red,B_red,p,r


    
def dmd_trunc_svd_with_var_cutoff(x0,x1,var_cutoff):
    ''' x1= A x0
    A= x1 *pinv(x0)= x1* v_bar Sigma^(-1) Transpose(U_bar)
     choose the number of components in U_bar so that it explains
     var_cutoff ratio of the variance
    '''
    assert x0.shape ==x1.shape
    U,sigma,V=scipy.linalg.svd(x0,full_matrices=False)
    exp_var_ratio=np.cumsum(sigma**2)/np.sum(sigma**2)
    n_components=np.searchsorted(exp_var_ratio, var_cutoff)+1
    U_bar=U[:,:n_components]
    sigma_bar=sigma[:n_components]
    V_bar=V[:n_components]
    A_bar=np.linalg.multi_dot((x1,V_bar.T,np.diag(1/sigma_bar),U_bar.T))
    A_red=np.linalg.multi_dot((U_bar.T,x1,V_bar.T,np.diag(1/sigma_bar)))
    return A_bar,A_red,U_bar,sigma,V_bar

def dmd_trunc_svd_with_rank_cutoff(x0,x1,n_components):
    ''' x1= A x0
    A= x1 *pinv(x0)= x1* v_bar Sigma^(-1) Transpose(U_bar)
     choose the number of components in U_bar so that it explains
     var_cutoff ratio of the variance
    '''
    assert x0.shape ==x1.shape
    U,sigma,V=scipy.linalg.svd(x0,full_matrices=False)
    U_bar=U[:,:n_components]
    sigma_bar=sigma[:n_components]
    V_bar=V[:n_components]
    A_bar=np.linalg.multi_dot((x1,V_bar.T,np.diag(1/sigma_bar),U_bar.T))
    A_red=np.linalg.multi_dot((U_bar.T,x1,V_bar.T,np.diag(1/sigma_bar)))
    return A_bar,A_red,U_bar,sigma,V_bar

def delay_dmd_with_control_with_var_cutoff(X0,X1,Y,var_cutoff):
    
    n=X0.shape[0]
    assert X0.shape==X1.shape and X0.shape[1]==Y.shape[1]
    
    Omega=np.vstack((X0,Y))
    
    U,Sigma,V=np.linalg.svd(Omega,full_matrices=False)
    
    exp_var_ratio=np.cumsum(Sigma**2)/np.sum(Sigma**2)
    p=np.searchsorted(exp_var_ratio, var_cutoff)+1
    
    U_bar1,Sigma_bar,V_bar=U[:n,:p],Sigma[:p],V[:p]
    
    U_bar2=U[n:,:p]
    
    A_bar=np.linalg.multi_dot((X1,V_bar.T,np.diag(1/Sigma_bar),U_bar1.T))
    B_bar=np.linalg.multi_dot((X1,V_bar.T,np.diag(1/Sigma_bar),U_bar2.T))
    
    
    eigenvals_X1,U_tilde=np.linalg.eig(X1@X1.T)
    
    exp_var_ratio_x1=np.cumsum(eigenvals_X1)/np.sum(eigenvals_X1)
    r=np.searchsorted(exp_var_ratio_x1, var_cutoff)+1
    
    
    U_tilde=U_tilde[:,:r]
    
    A_red=np.linalg.multi_dot((U_tilde.T,A_bar,U_tilde))
    
    B_red=np.matmul(U_tilde.T,B_bar)
    
    return A_bar,B_bar,U_tilde, A_red,B_red,p,r


    
def dmd_trunc_svd_with_var_cutoff(x0,x1,var_cutoff):
    ''' x1= A x0
    A= x1 *pinv(x0)= x1* v_bar Sigma^(-1) Transpose(U_bar)
     choose the number of components in U_bar so that it explains
     var_cutoff ratio of the variance
    '''
    assert x0.shape ==x1.shape
    U,sigma,V=scipy.linalg.svd(x0)
    exp_var_ratio=np.cumsum(sigma**2)/np.sum(sigma**2)
    n_components=np.searchsorted(exp_var_ratio, var_cutoff)+1
    U_bar=U[:,:n_components]
    sigma_bar=sigma[:n_components]
    V_bar=V[:n_components]
    A_bar=np.linalg.multi_dot((x1,V_bar.T,np.diag(1/sigma_bar),U_bar.T))
    A_red=np.linalg.multi_dot((U_bar.T,x1,V_bar.T,np.diag(1/sigma_bar)))
    return A_bar,A_red,U_bar,sigma_bar,V_bar


def dmd_trunc_svd_with_dim_cutoff(x0,x1,n_components):
    ''' x1= A x0
    A= x1 *pinv(x0)= x1* v_bar Sigma^(-1) Transpose(U_bar)
     choose the number of components in U_bar so that it explains
     var_cutoff ratio of the variance
    '''
    assert x0.shape ==x1.shape
    U,sigma,V=scipy.linalg.svd(x0)
    
    U_bar=U[:,:n_components]
    sigma_bar=sigma[:n_components]
    V_bar=V[:n_components]
    A_bar=np.linalg.multi_dot((x1,V_bar.T,np.diag(1/sigma_bar),U_bar.T))
    A_red=np.linalg.multi_dot((U_bar.T,x1,V_bar.T,np.diag(1/sigma_bar)))
    return A_bar,A_red,U_bar,sigma_bar,V_bar

    
    
    
        
    
    
        