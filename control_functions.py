#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 13:56:10 2024

@author: anandita
"""

import numpy as np
#import control as ct
from scipy.linalg import expm
from joblib import Parallel, delayed
import scipy


def compute_matrix_power(A,B,t):
    return np.linalg.matrix_power(A,t)@B


def ct_matrix(A,B,T):
    #ct=np.hstack([np.linalg.matrix_power(A,t)@B for t in range(T+1)])
    ct=np.hstack(Parallel(n_jobs=-1)(delayed(compute_matrix_power)(A,B,t)  for t in range(T)))
    return ct
                 
def discrete_ct_gramian_at_T(A,B,T):
    ct=ct_matrix(A,B,T)
    W_ct=np.matmul(ct,ct.T)
    return W_ct

def min_energy_u_at_t(ct,W_ct,A,B,t,xf,x0):
    
    u_min_energy=np.linalg.multi_dot((ct.T, np.linalg.pinv(W_ct),(xf- np.linalg.matrix_power(A,t)@x0)))
    return u_min_energy

# def min_energy_u_for_all_t(ct,A,B,T,xf,x0):
#     ut_min_energy=Parallel(n_jobs=-1)(delayed(min_energy_u_at_t)(ct,A,B,t,xf,x0) for t in range(1,T+1))
    
#     return (np.array(ut_min_energy)).T
    
def min_energy_u_gramian_alternate_at_t(ct,A,B,t,xf,x0):
    m=B.shape[1]
    u_min_energy=np.linalg.multi_dot((np.linalg.pinv(ct,1e-10),(xf- np.linalg.matrix_power(A,t)@x0)))
    return u_min_energy

# def min_energy_u_gramian_alternate_all_t(ct,A,B,T,xf,x0):
#     u_min_energy=Parallel(n_jobs=-1)(delayed(min_energy_u_gramian_alternate_at_t)(ct,A,B,t,xf,x0) for t in range(1,T+1))
#     return np.array(u_min_energy).T

    
def minimum_energy_input_at_t_cont(A,B,W,T,t,x_f,x0):
    u_t=np.linalg.multi_dot((np.transpose(B),expm(np.transpose(A*(T-t))),np.linalg.inv(W),
                              (x_f-np.matmul(expm(T*A),x0))))
    return u_t

def minimum_energy_input_cont(A,B,W,tf,time_array,xf,x0):
    u_t=np.array(Parallel(n_jobs=-1)(delayed(minimum_energy_input_at_t_cont)(A, B, W, tf, ti, xf,x0) for ti in time_array)).T
    return u_t

def minimum_energy_input_discrete_at_t(A,B,W,T,t,x_f,x0):
    u_t=np.linalg.multi_dot((np.transpose(B),np.linalg.matrix_power(np.transpose(A),(T-t-1)),np.linalg.inv(W),(x_f-np.linalg.matrix_power(A, T)@x0)))
    return u_t

def minimum_energy_input_discrete(A,B,W,T,xf,x0):
    time_array=np.arange(T)
    u_t=np.array(Parallel(n_jobs=-1)(delayed(minimum_energy_input_discrete_at_t)(A, B, W, T, ti, xf,x0) for ti in time_array)).T
    return u_t
    


def discrete_finite_time_grammian(time_array,A,B):
    ''' computing finite time grammian for discrete time '''
    W_c=np.zeros(A.shape)
    for ti in time_array:
        C_ti=np.matmul(np.linalg.matrix_power(A, ti),B)
        C_2=np.matmul(C_ti,np.transpose(C_ti))
        W_c=W_c+C_2
    return W_c

def continuous_finite_time_grammian(time_array,A,B,dt):
    ''' computing finite time grammian for continuous time '''
    W_c=np.zeros(A.shape)
    for ti in time_array:
        C_ti=np.matmul(expm(A* ti),B)
        C_2=dt*np.matmul(C_ti,np.transpose(C_ti))
        W_c=W_c+C_2
    return W_c
    

def out_strength(A):
    '''
    Input: Connectivity/adjacency matrix of the network
    Output: Dictionary of number of connections/edges going out from each node
    '''
    return dict(enumerate(np.linalg.norm(A,ord=1,axis=1)))

def in_strength(A):
    '''
    Input: Connectivity/adjacency matrix of the network
    Output: Dictionary of number of connections/edges going into each node
    '''
    return dict(enumerate(np.linalg.norm(A,ord=0,axis=0)))

def ratio_out_strength_in_strength(A):
    '''
    Parameters
    ----------
    A : Connectivity matrix

    Returns
    -------
    r: Ratio of the out strength and in strength described above for each node

    '''
    out_strength_array=np.array(list(out_strength(A)))
    in_strength_array=np.array(list(in_strength(A)))
    r=np.divide(out_strength_array,in_strength_array)
    return dict(enumerate(r))


def controllability_gramian_i(A,B,i):
    '''

    Parameters
    ----------
    A : Connectivity matrix
    B : Control input matrix
    i : identity of the node for which to calculate the controllability Grammian

    Returns
    -------
    W_i: Controllability Gramian corresponding to using node i as driver

    '''
    b_i=B[:,i]
    Q=np.outer(b_i,b_i)
    W_i=ct.lyap(A,Q)
    return W_i


def controllability_gramian_by_node(A,B):
    '''
    

    Parameters
    ----------
    A : Connectivity matrix
    B : Control input matrix

    Returns
    -------
    ct_gram_by_node: Dictionary of controllability Grammians by node 

    '''
    n=len(B[0])
    
    ct_gram_by_node=[controllability_gramian_i(A, B,i) 
                                    for i in range(n)]
    return ct_gram_by_node


def observability_gramian_i(A,C,i):
    '''
    

    Parameters
    ----------
    A : Connectivity matrix
    C : Observability matrix
    i : identity of the node for which to calculate the observability matrix

    Returns
    -------
    M_i: Observability Gramian for the node i

    '''
    c_i=C[:,i]
    Q=np.outer(c_i,c_i)
    M_i=ct.lyap(A.T,Q)
    return M_i

def observability_gramian_by_node(A,B):
    '''
    

    Parameters
    ----------
    A : Connectivity matrix
    C: Observability matrix

    Returns
    -------
    obs_gram_by_node: Dictionary of observability Grammians by node 

    '''
    n=len(B[0])
    obs_gram_by_node=dict(enumerate([observability_gramian_i(A, B, i) for i in range(n)]))
    return obs_gram_by_node


def lindmark_centrality_by_node(A,B,C):
    '''
    

    Parameters
    ----------
    A : Connectivity matrix
    B : Control input matrix
    C : Observability matrix

    Returns
    -------
    r: dictionary of Lindmark centrality by node, r_i=Tr(W_i)/Tr(M_i)

    '''
    n=len(B[0])
    ct_gram_by_node=controllability_gramian_by_node(A, B)
    obs_gram_by_node=observability_gramian_by_node(A, C)
    r_i=[np.trace(ct_gram_by_node[i])/np.trace(obs_gram_by_node[i]) for i in range(n)]
    r=dict(enumerate(r_i))
    return r

def driver_centrality(A,B):
    '''
    Parameters
    ----------
    A : Connectivity matrix
    B : Control input matrix

    Returns
    -------
    Dictionary of driver centrality for each node

    '''
    n=len(B[0])
    ct_gram_by_node=controllability_gramian_by_node(A, B)
    driver_centrality_by_node=[(1/n)*np.sum(1/np.diag(ct_gram_by_node[i])) for i in range(n)]
    return dict(enumerate(driver_centrality_by_node))


def target_centrality(A,B):
    '''
    Parameters
    ----------
    A : Connectivity matrix
    B : Control input matrix

    Returns
    -------
    Dictionary of driver centrality for each node

    '''
    n=len(B[0])
    ct_gram_by_node=controllability_gramian_by_node(A, B)
    target_centrality_by_node=[(1/n)*np.sum(np.array([1/(ct_gram_by_node[i][j,j]) for j in range(n)])) for i in range(n)]
    return dict(enumerate(target_centrality_by_node))


def trace_inv_gram(Wc):
    n=Wc.shape[0]
    if np.linalg.matrix_rank(Wc)==n:
        return np.trace(np.linalg.inv(Wc))
    elif np.linalg.matrix_rank(Wc)<n:
        return np.trace(np.linalg.pinv(Wc))

def log_det_gram(eigenvals_Wc):
    eigenvals_Wc_nonzero=eigenvals_Wc[eigenvals_Wc>0]
    return np.sum(np.log(eigenvals_Wc_nonzero))

def control_mat(u_0_t_1,x_0_t_1,x_1_t):
    '''
    

    Parameters
    ----------
    u_0_t_1 : The control inputs u from time 0 to t-1, each
              column is a timestep
    x_0_t_1 : The network states x from time 0 to t-1, each 
             column is a timestep
    x_1_t : The network stats from time 1 to t, each column 
            is a timestep

    Returns
    -------
    The dynamic matrix of shape N times M+N, the first M 
    columns correspond to the matrix input matrix B and
    the last N columns correspond to A

    '''
    #asserting that the number of columns for u corresponding
    #to the number of timesteps is the same as the number of
    #timesteps as x_0_t_1
    assert u_0_t_1.shape[1]==x_0_t_1.shape[1]
    u_x_t_1=np.vstack((u_0_t_1,x_0_t_1))
    
    return np.matmul(x_1_t,np.linalg.pinv(u_x_t_1))

    
def discrete_RNN(x0,T,n,phi,J):
    x_t=np.zeros((n,T))
    x_t[:,0]=x0
    
    for i in range(1,T):
        x_t[:,i]=np.matmul(J,phi(x_t[:,i-1]))
    return x_t

##########################################################################
#Data driven control functions
from scipy.linalg import pinv, null_space, svd, eigh

def CT_zero_init_cond(Xf,U,m,T,eps=1e-10):
    '''Returns the controllability Gramian for zero initial condition'''
    if np.linalg.matrix_rank(U) != m*T:
        print('Rank condition not satisfied')
        return None
    CT=Xf@pinv(U,eps)
    return CT

def cont_gram_data_driven_zero_init_cont(Xf,U,m,T,eps=1e-10):
    if np.linalg.matrix_rank(U) == m*T:
       CT=CT_zero_init_cond(Xf, U, m, T)
       W=CT@CT.T
       return W

def eigenvecs_and_eigenvals_zero_init_cond_cont_gram(Xf,U,m,T,eps=1e-10):
    if np.linalg.matrix_rank(U) == m*T:
        W=cont_gram_data_driven_zero_init_cont(Xf,U,m,T,eps=1e-10)
        eigenvals_W_dd,eigenvecs_W_dd=np.linalg.eig(W)
        eigenvals_W_dd=np.flip(np.sort(eigenvals_W_dd.real))
        eigenvecs_W_dd=eigenvecs_W_dd[:,np.flip(np.argsort(eigenvals_W_dd.real))].real
        
        return eigenvals_W_dd,eigenvecs_W_dd

        
def min_energy_control_zero_init_cond(pinv_CT,xf,eps=1e-10):
    min_energy_u=pinv_CT@xf
    return min_energy_u

def coef_approx_min_energy_u_zero_init_cond(Xf,U,eps=1e-10):
    coef=U@pinv(Xf,eps)
    return coef
    
def approx_min_energy_control_zero_init_cond(coef,xf):
   return coef @xf

def pred_xf_from_zero_init_cond(Xf,U,m,T,u,eps=1e-10):
    if np.linalg.matrix_rank(U) == m*T:
       CT=CT_zero_init_cond(Xf, U, m, T)
       xf=CT@u
       return xf
    else:
        print("Rank condition not satisfied")

def compute_matrices_for_ddc(X0, Xf, U, m, T, eps=1e-6):
    S = np.vstack((X0,Xf))
    
    KU = null_space(U,eps)
    K0 = null_space(X0,eps)
    K = np.hstack((KU,K0))
    
    SK = S@K
    # try:
    #     K_SK = null_space(SK,eps)
    # except np.linalg.LinAlgError:
    #     K_SK=null_space(SK,eps,lapack_driver='gesvd')
    
    return S,KU,K0,K,SK
    
       
def CT_data_driven_nonzero_ic(X0, Xf,K0, U, m, T, eps=1e-6):
    n = X0.shape[0]
    # Check rank
    rank_condition = np.linalg.matrix_rank( np.vstack([X0,U]) )
    if not rank_condition == (m*T + n):
        print(f'WARNING: rank condition not satisfied. rank={rank_condition}')
    
    Ctrb=Xf@K0 @pinv(U@K0,eps)
    return Ctrb

def coefs_for_min_energy_u_nonzero_init_cond(X0,K0,KU,K,SK,K_SK, Xf,U, m, T,n, eps=1e-10):
    # Check rank
    rank_condition = np.linalg.matrix_rank( np.vstack([X0,U]) )
    if not rank_condition == (m*T + n):
        print(f'WARNING: rank condition not satisfied. rank={rank_condition}')
    
    d_U=KU.shape[1]
    coeffirst =U @ K0@ pinv(SK,eps)[d_U:]
    
    coefsecond =U @ K0 @ K_SK[d_U:]@ pinv(U@ K0 @K_SK[d_U:],eps) @U@K0 @pinv(SK,eps)[d_U:]
    
    return coeffirst,coefsecond


def min_energy_u_nonzero_init_cond(coef_first,coef_second,x0,xf, U, m, T,Xf,K0, eps=1e-6,terms='first'):
    XfK0=Xf@K0

    # reachability of xf
    a, residuals_c, rank_c, s = np.linalg.lstsq(XfK0, xf, rcond=None)
    if len(residuals_c)>0: print(f'rank(X_f@K_0)={rank_c},not reachable,res={residuals_c[0]}') 
    #print(f'rank(X_f@K_0)={rank_c},reachable,res={residuals_c}') if len(residuals_c)==0  else print(f'rank(X_f@K_0)={rank_c},not reachable,res={residuals_c}')
    
    #projecting xf if xf is not reachable
    if len(residuals_c)>0:
        xf=XfK0@pinv(XfK0)@xf
    
    
    # Create matrix
    if np.ndim(x0)==1 and np.ndim(xf)==1:
        z = np.concatenate((x0,xf))
    if np.ndim(x0)==2 and np.ndim(xf)==2:
        z = np.vstack((x0,xf))
    
    if terms=='first':
        u_opt_dd= coef_first@z
    elif terms=='all':
        u_opt_dd=(coef_first-coef_second)@z
    
    return u_opt_dd
    
    


def control_signal_without_x0(X0, Xf,xf, U, m, T, eps=1e-6):
    n = X0.shape[0]
    
    KU = null_space(U)
    K0 = null_space(X0)
    K = np.hstack((KU,K0))
    
    XfK= Xf@K
    K_XfK= null_space(XfK)
    
    # Check rank
    rank_condition = np.linalg.matrix_rank( np.vstack([X0,U]) )
    if not rank_condition == (m*T + n):
        print(f'WARNING: rank condition not satisfied. rank={rank_condition}')
    
    # Compute optimal control
    gammafirst = pinv(XfK,eps)@xf
    try:
        gammasecond = K_XfK@ pinv(U@K@K_XfK,eps) @U@K @pinv(XfK,eps)@xf
        gamma = gammafirst-gammasecond
    except:
        gamma = gammafirst
        
    u_opt_dd = U @ K @ gammafirst
    
    return u_opt_dd

def xf_pred_using_CT_nonzero_ic(X0, Xf, U, m, T,u, eps=1e-10):
    CT_dd_nonzero_ic=CT_data_driven_nonzero_ic(X0, Xf, U, m, T)
    
    xf=CT_dd_nonzero_ic@u
    
    return xf

def x0_u_coef_for_xf_pred(X0,Xf,K0,KU,U,m,T,eps=1e-10):
    
    coef_x0=Xf@KU@pinv(X0@KU,eps)
    
    coef_u=Xf@K0@pinv(U@K0,eps)
    
    return coef_x0,coef_u
    
    
def xf_pred_dd_using_alpha_and_beta(coef_x0,x0,coef_u,u):
    '''
    x0= X0@KU@ alpha
    u=U@K0@ beta
    '''
    
    
    xf=coef_x0@x0+coef_u@u
    
    return xf

def coef_for_u_alternate(X0,Xf,U,eps=1e-10):
    X1=np.vstack((X0,Xf))
    X2=np.vstack((X0,U))
    try:
        coef_u_alternate=pinv(X1 @ pinv(X2,eps),eps)
    except np.linalg.LinAlgError:
        a1=X1@pinv(X2,eps)
        m,n=a1.shape
        u,s,v=scipy.linalg.svd(a1,full_matrices=False,lapack_driver='gesvd')
        sn=np.count_nonzero((s>1e-10).astype(int))
        sigma = np.zeros((m, n))
        for i in range(sn):
            sigma[i, i] = 1/s[i]
        coef_u_alternate=(u@sigma@v).T
    return coef_u_alternate
    
    
def u_opt_from_alternate_formula(coef_u_alternate,x0,xf,eps=1e-10):
    n=len(x0)
    z=np.concatenate((x0,xf))
    u=(coef_u_alternate@z)[n:]
    
    return u

def xf_from_u_alternate_formula(X0,Xf,U,x0,u,m,T,eps=1e-10):
    n = X0.shape[0]
    rank_condition = np.linalg.matrix_rank( np.vstack([X0,U]) )
    if not rank_condition == (m*T + n):
        print(f'WARNING: rank condition not satisfied. rank={rank_condition}')
    
    
    X1=np.vstack((X0,Xf))
    X2=np.vstack((X0,U))
    z1=np.concatenate((x0,u))
    
    xf=((X1 @ pinv(X2,eps),eps)@z1)[n:]
    
    return xf

def xf_from_u_approx_dd(U,X0,Xf,u,eps=1e-12):
    n=Xf.shape[0]
    S = np.vstack((X0,Xf))
    
    KU = null_space(U)
    K0 = null_space(X0)
    K = np.hstack((KU,K0))
    
    SK = S@K
    
    xf=(pinv(U@K @pinv(SK,eps),eps)@u)[n:]
    
    return xf

  