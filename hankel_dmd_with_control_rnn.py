#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 09:44:52 2024

@author: anandita
"""

import numpy as np
import dmd_functions as dmdf
import matplotlib.pyplot as plt
import control_functions as cf
import connectivity_matrix_functions as cmf
from sklearn.metrics import r2_score
import time

#number_of_neurons
n=100

np.random.seed(26)
A=np.random.normal(0,0.2,size=(n,n))

offset=100
T_without_input=10000
T_with_input=10000
T_test=20

#checking if eDMD can predict dynamics for RNN
x0=np.random.normal(0,1,size=n)
#running dynamics without input
X_without_input=cmf.discrete_RNN_1(x0, offset+T_without_input, n, np.tanh, A)
X_without_input=X_without_input[:,offset:]

L=9
var_cutoff=0.95

#dmd on X without inputs
# t0=time.time()
# X_delay_without_input=dmdf.delay_coordinates(X_without_input, L)

# A_bar_0,A_red_0,U_bar_0,sigma_0,V_bar_0=dmdf.dmd_trunc_svd_with_var_cutoff(X_delay_without_input[:,:-1],X_delay_without_input[:,1:],var_cutoff)
# print('Time taken to calculate delay dmd on X',time.time()-t0)

#fitting edmd with control for rnn
#Y=np.random.binomial(1,0.02,size=(m,T-1))
m=20

B=np.zeros((n,m))

B[np.random.choice(n,size=m,replace=False),np.arange(m)]=1

# Y=np.zeros((m,T-1+L))

# for i in range(T-1+L):
#     Y[np.random.choice(m,size=2,replace=False),i]=1
#Y=np.random.normal(0,1,size=(m,T_with_input+L-1))
Y=np.array([np.sin(np.random.uniform(0,1)*np.arange(T_with_input+L-1)) for _ in range(m)])

x0_input=X_without_input[:,:-1]

X_with_inputs=cmf.discrete_RNN_with_inputs_1(x0, T_with_input+L, n, np.tanh, A, B, Y)

X_with_inputs_delay=dmdf.delay_coordinates(X_with_inputs, L)


t0=time.time()
#delay dmd with control on X with inputs
A_bar_u,B_bar_u,U_tilde_u, A_red_u,B_red_u,p_u,r_u=dmdf.dmd_with_control_with_var_cutoff(X_with_inputs_delay,Y[:,:T_with_input-1],var_cutoff)
print('Time taken to calculate delay dmd with control on X with inputs',time.time()-t0)

#delay dmd on X with inputs
# t0=time.time()
# A0_bar_u,A0_red_u,U0_bar_u,sigma_0u,V0_bar_u=dmdf.dmd_trunc_svd_with_var_cutoff(X_with_inputs_delay[:,:-1],X_with_inputs_delay[:,1:],var_cutoff)
# print('Time taken to compute delay dmd for X with inputs',time.time()-t0)


X_lift=U_tilde_u.T@X_with_inputs_delay
t0=time.time()
C=X_with_inputs[:,L:]@np.linalg.pinv(X_lift,rcond=1e-6)
print('Time taken to calculate C',time.time()-t0)

##################################################################3
timesteps_to_predict=5
n_runs=100
# r2_score_X_pred_red_and_X_true_A0=np.zeros((n_runs,timesteps_to_predict))
# r2_score_X_pred_actual_and_X_true_A0=np.zeros((n_runs,timesteps_to_predict))

r2_score_X_pred_red_and_X_true_Au=np.zeros((n_runs,timesteps_to_predict))
#r2_score_X_pred_actual_and_X_true_Au=np.zeros((n_runs,timesteps_to_predict))
r2_score_X_pred_actual_and_X_true_Au_1=np.zeros((n_runs,timesteps_to_predict))

# r2_score_X_pred_red_and_X_true_Au0=np.zeros((n_runs,timesteps_to_predict))
# r2_score_X_pred_actual_and_X_true_Au0=np.zeros((n_runs,timesteps_to_predict))


for run_i in range(n_runs):
    t=np.random.randint(500,1000)
    #prediction
    x0=X_without_input[:,t]
    
    #inputs
    Y=np.array([np.cos(np.random.uniform(0,1)*np.arange(timesteps_to_predict+L)) for _ in range(m)])

    #ground_truth full space
    X_test_with_inputs=cmf.discrete_RNN_with_inputs_1(x0, timesteps_to_predict+L+1, n, np.tanh, A, B, Y)

    X_test_with_inputs_delay=dmdf.delay_coordinates(X_test_with_inputs, L)

    x0_test_with_inputs_delay=X_test_with_inputs_delay[:,0]
    #predictions using Au
    #predicted full space
    X_pred_with_inputs_delay_Au=cmf.lti_discrete(A_bar_u,B_bar_u,Y,timesteps_to_predict+1,x0_test_with_inputs_delay)

    #print('FVE true by predicted full space', r2_score(X_test_with_inputs_delay,X_pred_with_inputs_delay,multioutput='raw_values'))
    #r2_score_X_pred_actual_and_X_true_Au[run_i]=r2_score(X_test_with_inputs_delay[-n:,1:].real,X_pred_with_inputs_delay_Au[-n:,1:].real,multioutput='raw_values')

    #prediction in reduced space 
    x0_proj_from_delay_Au=np.matmul(U_tilde_u.T,x0_test_with_inputs_delay)

    X_test_proj_from_delay_Au=np.matmul(U_tilde_u.T,X_test_with_inputs_delay)

    X_proj_pred_Au=cmf.lti_discrete(A_red_u,B_red_u,Y[:,L:],timesteps_to_predict+1,x0_proj_from_delay_Au)
    
    r2_score_X_pred_red_and_X_true_Au[run_i]=r2_score(X_test_proj_from_delay_Au[:,1:],X_proj_pred_Au[:,1:],multioutput='raw_values')
    
    r2_score_X_pred_actual_and_X_true_Au_1[run_i]=r2_score(X_test_with_inputs_delay[-n:,1:].real, C@X_proj_pred_Au[:,1:],multioutput='raw_values')
    # #predictions A0
    # #predicted full space
    # X_pred_with_inputs_delay_A0=cmf.lti_without_control(A_bar_0,A_bar_0.shape[0],timesteps_to_predict+1,x0_test_with_inputs_delay)

    # #print('FVE true by predicted full space', r2_score(X_test_with_inputs_delay,X_pred_with_inputs_delay,multioutput='raw_values'))
    # r2_score_X_pred_actual_and_X_true_A0[run_i]=r2_score(X_test_with_inputs_delay[-n:,1:].real,X_pred_with_inputs_delay_A0[-n:,1:].real,multioutput='raw_values')

    # #prediction in reduced space
    # x0_proj_from_delay_A0=np.matmul(U_bar_0.T,x0_test_with_inputs_delay)

    # X_test_proj_from_delay_A0=np.matmul(U_bar_0.T,X_test_with_inputs_delay)

    # X_proj_pred_A0=cmf.lti_without_control(A_red_0,A_red_0.shape[0],timesteps_to_predict+1,x0_proj_from_delay_A0)
    
    # r2_score_X_pred_red_and_X_true_A0[run_i]=r2_score(X_test_proj_from_delay_A0[:,1:],X_proj_pred_A0[:,1:],multioutput='raw_values')
    
    # #predictions Au0
    # #predicted full space
    # X_pred_with_inputs_delay_Au0=cmf.lti_without_control(A0_bar_u,A0_bar_u.shape[0],timesteps_to_predict+1,x0_test_with_inputs_delay)

    # #print('FVE true by predicted full space', r2_score(X_test_with_inputs_delay,X_pred_with_inputs_delay,multioutput='raw_values'))
    # r2_score_X_pred_actual_and_X_true_Au0[run_i]=r2_score(X_test_with_inputs_delay[-n:,1:].real,X_pred_with_inputs_delay_Au0[-n:,1:].real,multioutput='raw_values')

    # #prediction in reduced space
    # x0_proj_from_delay_Au0=np.matmul(U0_bar_u.T,x0_test_with_inputs_delay)

    # X_test_proj_from_delay_Au0=np.matmul(U0_bar_u.T,X_test_with_inputs_delay)

    # X_proj_pred_Au0=cmf.lti_without_control(A0_red_u,A0_red_u.shape[0],timesteps_to_predict+1,x0_proj_from_delay_Au0)
    
    # r2_score_X_pred_red_and_X_true_Au0[run_i]=r2_score(X_test_proj_from_delay_Au0[:,1:],X_proj_pred_Au0[:,1:],multioutput='raw_values')
    
   
###################################################################################################    

#r2 scores for proj space
plt.figure()
plt.plot(np.mean(r2_score_X_pred_red_and_X_true_Au,axis=0),marker='.',label='Au')
plt.fill_between(np.arange(timesteps_to_predict),np.min(r2_score_X_pred_red_and_X_true_Au,axis=0),
                  np.max(r2_score_X_pred_red_and_X_true_Au,axis=0),alpha=0.4)

# plt.plot(np.mean(r2_score_X_pred_red_and_X_true_A0,axis=0),marker='.',label='A0')
# plt.fill_between(np.arange(timesteps_to_predict),np.min(r2_score_X_pred_red_and_X_true_A0,axis=0),
#                   np.max(r2_score_X_pred_red_and_X_true_A0,axis=0),alpha=0.4)

# plt.plot(np.mean(r2_score_X_pred_red_and_X_true_Au0,axis=0),marker='.',label='Au0')
# plt.fill_between(np.arange(timesteps_to_predict),np.min(r2_score_X_pred_red_and_X_true_Au0,axis=0),
#                   np.max(r2_score_X_pred_red_and_X_true_Au0,axis=0),alpha=0.4)

plt.legend()
plt.xlabel('timestep predicted')
plt.ylabel('R2 score between x_pred and x_true')
plt.title('R2 score between pred and observed in proj space')
plt.show()
plt.close()



plt.figure()
# plt.plot(np.mean(r2_score_X_pred_actual_and_X_true_Au,axis=0),marker='.',label='Au')
# plt.fill_between(np.arange(timesteps_to_predict),np.min(r2_score_X_pred_actual_and_X_true_Au,axis=0),
#                   np.max(r2_score_X_pred_actual_and_X_true_Au,axis=0),alpha=0.4)

plt.plot(np.mean(r2_score_X_pred_actual_and_X_true_Au_1,axis=0),marker='.',label='Au')
plt.fill_between(np.arange(timesteps_to_predict),np.min(r2_score_X_pred_actual_and_X_true_Au_1,axis=0),
                  np.max(r2_score_X_pred_actual_and_X_true_Au_1,axis=0),alpha=0.4)


# plt.plot(np.mean(r2_score_X_pred_actual_and_X_true_A0,axis=0),marker='.',label='A0')
# plt.fill_between(np.arange(timesteps_to_predict),np.min(r2_score_X_pred_actual_and_X_true_A0,axis=0),
#                   np.max(r2_score_X_pred_actual_and_X_true_A0,axis=0),alpha=0.4)

# plt.plot(np.mean(r2_score_X_pred_actual_and_X_true_Au0,axis=0),marker='.',label='Au0')
# plt.fill_between(np.arange(timesteps_to_predict),np.min(r2_score_X_pred_actual_and_X_true_Au0,axis=0),
#                   np.max(r2_score_X_pred_actual_and_X_true_Au0,axis=0),alpha=0.4)

plt.xlabel('timestep predicted')
plt.ylabel('R2 score between x_pred and x_true')
plt.title('R2 score between pred  and observed in neuron space')
plt.show()
plt.close()

###############################################################
#calculating the Gramian and the minimum energy input
control_horizon=20
time_array=np.arange(control_horizon)

t0=time.time()
finite_time_Gramian=cf.discrete_finite_time_grammian(time_array,A_red_u,B_red_u)
print('Time taken to calculate finite time Gramian',time.time()-t0)

eigenvals_W,eigenvecs_W=np.linalg.eig(finite_time_Gramian)
eigenvecs_W=eigenvecs_W[:,np.flip(np.argsort(eigenvals_W.real))]
eigenvals_W=np.flip(np.sort(eigenvals_W.real))


# eigenvals_A_red,_=np.linalg.eig(A_red_u)

# fig,ax=plt.subplots()
# ax.scatter(eigenvals_A_red.real,eigenvals_A_red.imag)
# ax.set_aspect('equal')
# plt.show()
# plt.close()

#running the reduced linear system with the u_opt
red_space_dim=len(A_red_u)

#x0_red_space=np.random.normal(size=red_space_dim)
x0=X_without_input[:,t]
x0_red_space=np.linalg.pinv(C,rcond=1e-6)@x0

#choosing the eigenvec of W with largest eigenval to be the final
#state for the control problem


def flip_sign(v):
    n=len(v)
    r=np.count_nonzero((v<0).astype(int))
    if r>n/2:
        return -v
    else:
        return v

#xf_red_space=flip_sign(eigenvecs_W[:,1].real)

xf_red_space=np.linalg.pinv(C,rcond=1e-6)@X_with_inputs[:,t]

t0=time.time()
u_opt=cf.minimum_energy_input_discrete(A_red_u,B_red_u,finite_time_Gramian,control_horizon,xf_red_space,x0_red_space)
print('Time taken to calculate discrete min energy Gramian',time.time()-t0)

#u_opt=np.array([np.sin(np.random.uniform(0,1)*np.arange(control_horizon)) for _ in range(m)])

#simulating the linear system in the reduced space
xf_simulated=cmf.lti_discrete(A_red_u,B_red_u,u_opt,control_horizon+1,x0_red_space)

xf_simulated_proj=C@xf_simulated

plt.figure()
plt.plot(xf_red_space)
plt.plot(xf_simulated[:,-1])
#plt.title('Difference between simulated and target xf')
plt.show()
plt.close()

print('Distance between target xf and observed xf',np.linalg.norm(xf_red_space-xf_simulated[:,-1]))

#simulating the RNN with ut

xt_rnn_u_opt=cmf.discrete_RNN_with_inputs_1(x0,control_horizon+1,n,np.tanh,A,B,u_opt)

xf_rnn=C@xf_red_space

plt.figure()
plt.plot(xf_rnn)
plt.plot(xt_rnn_u_opt[:,-1]/np.max(xt_rnn_u_opt[:,-1]))
#plt.title('Difference between simulated and target xf')
plt.show()
plt.close()


print('Distance between target xf and observed xf',np.linalg.norm(xf_rnn-xt_rnn_u_opt[:,-1]))

r2_score_pred_and_obs=r2_score(xt_rnn_u_opt[1:],xf_simulated_proj[1:],multioutput='raw_values')


for i in range(5):
    plt.figure()
    plt.plot(xt_rnn_u_opt[i])
    plt.plot(xf_simulated_proj[i])
    plt.show()
    plt.close()











def cosine(v1,v2):
    return np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))
    

