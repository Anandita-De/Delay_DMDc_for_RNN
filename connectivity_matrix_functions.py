#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 10:02:54 2024

@author: anandita
"""

import numpy as np


def erdos_renyi_connectivity(p,w,N1,N2,seed=None):
    '''
    Creates the adjacency matrix for a random network where the 
    probability of connection between neuron i of population 1 with 
    N1 neurons and neuron j of population 2 with N2 neurons is p with
    weight w  and 0 with probability (1-p)

    Parameters
    ----------
    p : probability of connection between neuron i and neuron j
    w: weight of connection between neuron i and neuron j
    N1 : number of neurons in population 1
    N2 : number of neurons in population 2

    Returns
    -------
    W: Connectivity matrix of the network

    '''
    np.random.seed(seed)
    W=w*np.random.binomial(1,p,size=(N1,N2))
    return W


def EI_balanced_connectivity(p,we,wi,N_I,N_E,seed=None):
    '''
    Creates the weight matrix of a EI balanced network from the above
    Erdos Renyi network where the weights from the E neurons are we and
    weights from the I neurons are wi.

    Parameters
    ----------
    p : probability of connection between neuron i and neuron j
    we : weights from e neurons
    wi : weights from i neurons
    N_i : number of inhibitory neurons
    N_e : number of excitatory neurons
    Returns
    -------
    W: connectivity matrix of a balanced EI network

    '''
    wee=erdos_renyi_connectivity(p,we,N_E,N_E,seed)
    wei=erdos_renyi_connectivity(p,wi,N_E,N_I,seed+1)
    wie=erdos_renyi_connectivity(p,we,N_I,N_E,seed+2)
    wii=erdos_renyi_connectivity(p,wi,N_I,N_I,seed+3)

    we=np.hstack((wee,wei))
    wi=np.hstack((wie,wii))

    W=np.vstack((we,wi))
    W=W-np.mean(W,axis=1)[:,np.newaxis]
    return W

def EI_connectivity(p,we,wi,N_I,N_E,seed=None):
    '''
    Creates the weight matrix of a EI balanced network from the above
    Erdos Renyi network where the weights from the E neurons are we and
    weights from the I neurons are wi.

    Parameters
    ----------
    p : probability of connection between neuron i and neuron j
    we : weights from e neurons
    wi : weights from i neurons
    N_i : number of inhibitory neurons
    N_e : number of excitatory neurons
    Returns
    -------
    W: connectivity matrix of a balanced EI network

    '''
    wee=erdos_renyi_connectivity(p,we,N_E,N_E,seed)
    wei=erdos_renyi_connectivity(p,wi,N_E,N_I,seed+1)
    wie=erdos_renyi_connectivity(p,we,N_I,N_E,seed+2)
    wii=erdos_renyi_connectivity(p,wi,N_I,N_I,seed+3)

    we=np.hstack((wee,wei))
    wi=np.hstack((wie,wii))

    W=np.vstack((we,wi))
    return W

    
def real_eigenvecs_eigenvals(eigenvals,eigenvecs,sorting='real'):
    '''
    Given complex eigenvalues and eigenvectors, sorts the eigenvalues by
    the real parts then creates a real quasi-diagonal eigenvalue matrix with
    real eigenvalues on the diagonal and imaginary parts on the off diagonals.
    Also creates a real matrix with linear combinations of the eigenvecs.

    Parameters
    ----------
    eigenvals : complex eigenvalues
    eigenvecs : complex eigenvecs

    Returns
    -------
    quasi_diagonal : real matrix with sorted real part of eigenvalues on the
                     diagonal and the imaginary part on the off diagonal
    eigenvecs_real : Real matrix from linear combinations of eigenvectors from
                     the above quasi-diagonal eigenvalue matrix

    '''
    if sorting=='real':
        sorted_indices=np.argsort(eigenvals.real)
    elif sorting=='imag':
        sorted_indices=np.argsort(eigenvals.imag)
    elif sorting=='abs':
        sorted_indices=np.argsort(np.abs(eigenvals))
        
    eigenvals=eigenvals[sorted_indices]
    eigenvecs=eigenvecs[:,sorted_indices]
    N=len(eigenvals)
    quasi_diagonal=np.zeros((N,N))
    eigenvecs_real=np.zeros((N,N))
    i=0
    while i<N:
        ev=eigenvals[i]
        if ev.imag==0:
            quasi_diagonal[i,i]=ev.real
            eigenvecs_real[:,i]=eigenvecs[:,i].real
            i=i+1
        else:
            if (np.linalg.norm(eigenvecs[:,i].imag,ord=2))==0:
                print(np.linalg.norm(eigenvecs[:,i].imag,ord=2))
                print(i)
                break
            quasi_diagonal[i,i]=ev.real
            quasi_diagonal[i+1,i+1]=ev.real
            quasi_diagonal[i,i+1]=ev.imag
            quasi_diagonal[i+1,i]=-ev.imag
            eigenvecs_real[:,i]=eigenvecs[:,i].real
            eigenvecs_real[:,i+1]=eigenvecs[:,i].imag
            i=i+2
    return quasi_diagonal,eigenvecs_real

def real_schur_decomposition_with_sorted_evs(W,sorting='real'):
    '''
    Computes the real Schur decomposition of W, with sorted real parts of 
    the eigenvalues.

    Parameters
    ----------
    W : TYPE
        DESCRIPTION.

    Returns
    -------
    Q : Orthonnormal Schur basis
    U : Quasi upper triangular matrix

    '''
    
    eigenvals_w,eigenvecs_w=np.linalg.eig(W)
    quasidiag_w,eigenvecs_real_w=real_eigenvecs_eigenvals(eigenvals_w,eigenvecs_w,sorting)
    
    Q,R=np.linalg.qr(eigenvecs_real_w)
    
    U=np.linalg.multi_dot((R,quasidiag_w,np.linalg.inv(R)))
    
    return Q,U


def lti(A,B,u,N,dt,T,x0):
    time_array=np.arange(0,T,dt)
    x=np.zeros((N,len(time_array)))
    x[:,0]=x0
    for i in range(1,len(time_array)):
        x[:,i]=x[:,i-1]+ dt*(np.matmul(A,x[:,i-1])+np.matmul(B,u[:,i-1]))
    return x

def lti_discrete(A,B,u,T,x0):
    time_array=np.arange(0,T)
    N=len(x0)
    x=np.zeros((N,len(time_array)))
    x[:,0]=x0
    for i in range(1,len(time_array)):
        if np.ndim(u)==2:
            x[:,i]=(np.matmul(A,x[:,i-1])+np.matmul(B,u[:,i-1]))
        elif np.ndim(u)==1:
            x[:,i]=np.matmul(A,x[:,i-1])+(B*u[i-1]).reshape(N)
    return x

def lti_without_control(A,N,T,x0):
    time_array=np.arange(0,T)
    x=np.zeros((N,len(time_array)))
    x[:,0]=x0
    for i in range(1,len(time_array)):
        x[:,i]=x[:,i-1]+ (np.matmul(A,x[:,i-1]))
    return x


def linear_sys(A,T,x0):
    time_array=np.arange(0,T)
    N=len(x0)
    x=np.zeros((N,len(time_array)))
    x[:,0]=x0
    for i in range(1,len(time_array)):
        x[:,i]=np.matmul(A,x[:,i-1])
    return x

def discrete_RNN(x0,T,n,phi,J):
    x_t=np.zeros((n,T))
    x_t[:,0]=x0
    
    for i in range(1,T):
        x_t[:,i]=np.matmul(J,phi(x_t[:,i-1]))
    return x_t  

def discrete_RNN_1(x0,T,n,phi,J):
    x_t=np.zeros((n,T))
    x_t[:,0]=x0
    
    for i in range(1,T):
        x_t[:,i]=phi(np.matmul(J,x_t[:,i-1]))
    return x_t  

def discrete_RNN_with_inputs(x0,T,n,phi,J,B,ut,offset=0):
    x_t=np.zeros((n,T))
    x_t[:,0]=x0
    
    for i in range(1,T):
        if np.ndim(ut)==2:
            x_t[:,i]=np.matmul(J,phi(x_t[:,i-1]))+np.matmul(B,ut[:,i-1])
        elif np.ndim(ut)==1:
            x_t[:,i]=np.matmul(J,phi(x_t[:,i-1]))+(B*ut[i-1]).reshape(n)
    
    return np.array(x_t[:,offset:]) 

def discrete_RNN_with_inputs_1(x0,T,n,phi,J,B,ut,offset=0):
    x_t=np.zeros((n,T))
    x_t[:,0]=x0
    
    for i in range(1,T):
        if np.ndim(ut)==2:
            x_t[:,i]=phi(np.matmul(J,x_t[:,i-1])+np.matmul(B,ut[:,i-1]))
        elif np.ndim(ut)==1:
            x_t[:,i]=phi(np.matmul(J,x_t[:,i-1])+(B*ut[i-1]).reshape(n))
    
    return np.array(x_t[:,offset:]) 



def ricker(x0,T,J):
    n=int(len(x0))
    x_t=np.zeros((n,T))
    
    x_t[:,0]=x0
    
    for i in range(1,T):
        x_t[:,i]=x_t[:,i-1]*np.exp(np.diag(J)- np.matmul(J,x_t[:,i-1]))
    
    return x_t


def rnn_dynamics(x0,W,phi,T,dt):
    time_array=np.arange(0,T,dt)
    x_t=np.zeros((len(x0),len(time_array)))
    x_t[:,0]=x0
    for i in range(1,len(time_array)):
        x_t[:,i]=(1-dt)*x_t[:,i-1]+dt*phi(np.matmul(W,x_t[:,i-1]))
    
    return x_t

def rnn_dynamics_with_inputs(x0,W,phi,T,dt,ut):
    time_array=np.arange(0,T,dt)
    x_t=np.zeros((len(x0),len(time_array)))
    x_t[:,0]=x0
    for i in range(1,len(time_array)):
        x_t[:,i]=(1-dt)*x_t[:,i-1]+dt*phi(np.matmul(W,x_t[:,i-1])+ut[:,i-1])
    
    return x_t
##############################################################
#activation functions
def linear(x):
    return x

def relu(x):
    return x* ((x>0).astype(int))

def sigmoid(x):
    f= 1/(1+np.exp(-x))
    return f

def tanh(x):
    return np.tanh(x)

