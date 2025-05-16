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

#number_of_neurons

n=100

np.random.seed(26)
A=np.random.normal(0,0.2,size=(n,n))

T=5000

#fitting to 50 different trajectories with different initial conditions

X0_delay_list=[]
X1_delay_list=[]

subsample_size=30

random_subsample=np.random.choice(n,size=subsample_size,replace=False)

for in_co in range(20):
    #checking if eDMD can predict dynamics for RNN
    x0=np.random.normal(0,1,size=n)

    #getting timeseries for RNN dynamics
    X=cmf.discrete_RNN_1(x0, T, n, np.tanh, A)

    plt.figure()
    for i in range(5):
        plt.plot(X[i][1000:1100],label=i)
    plt.legend()
    plt.xlabel("time")
    plt.ylabel('activity')
    plt.show()
    plt.close()

    L=9
    Xt=X[:,4000:]
    Xt_subsample=Xt[random_subsample]
    X_delay=dmdf.delay_coordinates(Xt_subsample, L)
    X0_delay_list.append(X_delay[:,:-1])
    X1_delay_list.append(X_delay[:,1:])

X0_delay=np.hstack(X0_delay_list)
X1_delay=np.hstack(X1_delay_list)

var_cutoff=0.95

A_bar,A_red,U_bar,sigma,V_bar=dmdf.dmd_trunc_svd_with_var_cutoff(X0_delay,X1_delay,var_cutoff)


#plotting the variance explained in the delay space
plt.figure()
plt.plot(np.cumsum(sigma**2)/np.sum(sigma**2))
plt.title(f'FVE in delay space, L={L}')
plt.show()
plt.close()

X_lift=U_bar.T@X_delay

C=Xt[:,L:]@np.linalg.pinv(X_lift,rcond=1e-10)


######################################################################
#testing prediction
n_components=len(A_red)


timesteps_to_predict=10
n_runs=100
r2_score_X_pred_red_and_X_true_by_timebin=np.zeros((n_runs,timesteps_to_predict))
#r2_score_X_pred_actual_and_X_true_by_timebin=np.zeros((n_runs,timesteps_to_predict))
r2_score_X_pred_actual_and_X_true_1=np.zeros((n_runs,timesteps_to_predict))

r2_score_X_pred_red_and_X_true_by_components=np.zeros((n_runs,n_components))
#r2_score_X_pred_actual_and_X_true_by_neuron=np.zeros((n_runs,n))


for run_i in range(n_runs):
    #prediction
    x0=np.random.normal(0,1,size=n)
    
    #ground_truth full space
    X_t=cmf.discrete_RNN_1(x0, 2000, n, np.tanh, A)
    
    X_test=X_t[:,1000:1000+timesteps_to_predict+L+1]
    
    X_test_subsample=X_test[random_subsample]

    X_test_delay=dmdf.delay_coordinates(X_test_subsample, L)

    x0_test_delay=X_test_delay[:,0]
    
    X_pred_delay=cmf.linear_sys(A_bar,timesteps_to_predict+1,x0_test_delay)
    
    #r2_score_X_pred_actual_and_X_true_by_timebin[run_i]=r2_score(X_test_delay[-n:,1:],X_pred_delay[-n:,1:],multioutput='raw_values')
    
    #r2_score_X_pred_actual_and_X_true_by_neuron[run_i]=r2_score(X_test_delay[-n:,1:].T,X_pred_delay[-n:,1:].T,multioutput='raw_values')
    
    #prediction in reduced space
    x0_proj_from_delay=np.matmul(U_bar.T,x0_test_delay)

    X_test_proj_from_delay=np.matmul(U_bar.T,X_test_delay)

    X_proj_pred=cmf.linear_sys(A_red,timesteps_to_predict+1,x0_proj_from_delay)
    
    r2_score_X_pred_red_and_X_true_by_timebin[run_i]=r2_score(X_test_proj_from_delay[:,1:].real,X_proj_pred[:,1:],multioutput='raw_values')
    
    r2_score_X_pred_red_and_X_true_by_components[run_i]=r2_score(X_test_proj_from_delay[:,1:].T,X_proj_pred[:,1:].T,multioutput='raw_values')
    
    r2_score_X_pred_actual_and_X_true_1[run_i]=r2_score(X_test[:,(L+1):].real,C@X_proj_pred[:,1:],multioutput='raw_values')


#plotting a few observed and predicted trajectories in projected space
for i in range(5,10):
    plt.figure()
    plt.plot(X_test_proj_from_delay[i],label='obs')
    plt.plot(X_proj_pred[i],label='pred')
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Activity')
    plt.title(f'Predicted and observed in proj from delay space,{i}')
    plt.show()
    plt.close()

#plotting a few observed and predicted trajectories in neuron space using
#both prediction methods
for i in range(5,10):
    plt.figure()
    #plt.plot(X_test_delay[-n+i],marker='.',label='obs')
    plt.plot(X_test[i][-1-timesteps_to_predict:],marker='.',label='obs')
    #plt.plot(X_pred_delay[-n+i],marker='.',label='pred')
    plt.plot((C@X_proj_pred)[i],marker='^',label='pred 1')
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Activity')
    plt.title(f'Predicted and observed in activity space,{i} \n from delayDMD')
    plt.show()
    plt.close()

#plotting r2 score between observed and predicted in projected space
plt.figure()
plt.plot(np.arange(1,timesteps_to_predict+1),np.mean(r2_score_X_pred_red_and_X_true_by_timebin,axis=0),marker='.')
plt.fill_between(np.arange(1,timesteps_to_predict+1),np.min(r2_score_X_pred_red_and_X_true_by_timebin,axis=0),
                  np.max(r2_score_X_pred_red_and_X_true_by_timebin,axis=0),alpha=0.4)

plt.xlabel('timestep predicted')
plt.ylabel('R2 score between x_pred and x_true')
plt.title(f'R2 score between pred and observed in proj space,\n var_cutoff={var_cutoff}')
plt.show()
plt.close()


#plotting r2 score between observed and predicted in neuron space using last
#stack from delay space
# fig,ax=plt.subplots(figsize=(8,6))
# plt.plot(np.arange(1,timesteps_to_predict+1),np.mean(r2_score_X_pred_actual_and_X_true_by_timebin,axis=0),marker='.')
# plt.fill_between(np.arange(1,timesteps_to_predict+1),np.min(r2_score_X_pred_actual_and_X_true_by_timebin,axis=0),
#                   np.max(r2_score_X_pred_actual_and_X_true_by_timebin,axis=0),alpha=0.4)
# # ax.set_xticks([1,10])
# # ax.set_xticklabels([])
# # ax.set_yticks([-0.4,0.0,0.5,1.0])
# # ax.set_yticklabels([])
# # Customize tick lengths
# ax.tick_params(axis='x', length=6)
# ax.tick_params(axis='y', length=6)
# # Remove top and right spines
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)

# plt.xlabel('timestep predicted')
# plt.ylabel('R2 score between x_pred and x_true')
# plt.title(f'R2 score between pred and observed in real space,\n var_cutoff={var_cutoff}')
# #plt.savefig('plots/control_review_fig/r2_score_between_pred_and_observed_neuron_space.pdf',bbox_inches='tight',transparent=True)
# plt.show()
# plt.close()

#plotting r2 score between observed and predicted in neuron space using projection
#from reduced space
plt.figure()
plt.plot(np.arange(1,timesteps_to_predict+1),np.mean(r2_score_X_pred_actual_and_X_true_1,axis=0),marker='.')
plt.fill_between(np.arange(1,timesteps_to_predict+1),np.min(r2_score_X_pred_actual_and_X_true_1,axis=0),
                  np.max(r2_score_X_pred_actual_and_X_true_1,axis=0),alpha=0.4)

plt.xlabel('timestep predicted')
plt.ylabel('R2 score between x_pred and x_true')
plt.title(f'R2 score between pred and observed in real space,\n var_cutoff={var_cutoff}')
plt.show()
plt.close()

#plotting a few trajectories in neuron space
# fig,ax=plt.subplots(nrows=3,sharex=True,gridspec_kw = {'hspace':0},figsize=(8,6))
# for pi,i in enumerate(indices):
#     ax[pi].plot(X_t[i][1000+L+1+t-200-9:1000+L+1+t-9],color='k',linewidth=1)
#     ax[pi].axvline(x=200,color='blue',linestyle='--')
#     ax[pi].axvline(x=170,color='blue',linestyle='--')
#     ax[pi].set_axis_off()
# plt.tight_layout()
# plt.savefig('plots/control_review_fig/discrete_RNN_1_3_timeseries.pdf',bbox_inches="tight",transparent=True)
# plt.show()
# plt.close()

# #plotting trajectories predicted vs observed
# t=timesteps_to_predict
# indices=[95,2,72]

# fig,ax=plt.subplots(nrows=3,sharex=True,gridspec_kw = {'hspace':0},figsize=(8,6))
# for pi,i in enumerate(indices):
#     ax[pi].plot(np.arange(30),X_t[i][1000+L+1+t-30: 1000+L+1+t],color='k')
#     ax[pi].plot(np.arange(20,30),X_pred_delay[-n+i][1:],'--',color='deeppink')
#     ax[pi].set_axis_off()
#     ax[pi].axvline(x=20,color='blue',linestyle='--')
#     ax[pi].axvline(x=0,color='blue',linestyle='--')
# plt.tight_layout()
# plt.savefig('plots/control_review_fig/pred_and_observed_discrete_RNN_1_3_timeseries.pdf',bbox_inches="tight",transparent=True)
# plt.show()
# plt.close()













