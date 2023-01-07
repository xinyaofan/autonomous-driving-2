# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 07:01:52 2020

@author: qq429
"""
import os
import pandas as pd
import numpy as np

from sklearn.utils.validation import check_array



class DistanceMatrix():
    def __init__(self,A,B):
        self.A=A
        self.B=B
    
    def check_vec2matrix(self):
        if self.A.ndim==1:
            self.A=self.A[None]
        if self.B.ndim==1:
            self.B=self.B[None]
    
    def check_parwise(self):
        if self.A.shape != self.B.shape:
            print("Error: Pairwise failed!")
    
    def euclidean_dist_squared(self):
        """Computes the Euclidean distance between rows of 'A' and rows of 'B'
        
        Parameters
        ----------
        A : an N by D numpy array or 1 D array
        B: an T by D numpy array or 1 D array
        
        Returns: an array of size N by T containing the pairwise squared Euclidean distances.
        """
        self.check_vec2matrix()
    
        return np.sum(self.A**2, axis=1)[:,None] + np.sum(self.B**2, axis=1)[None] - 2 * np.dot(self.A,self.B.T)
 
    def euclidean_dist_squared_axis(self,axis=0):
        """Computes the Euclidean distance between 'A' and 'B' along spec axis
        
        
        Parameters
        ----------
        A : an N by D numpy array or 1D array
        B: an N by D numpy array or 1D array
        axis: 0 along the row, 1 along the column, default is along the column
        
        Returns: an array of size N by 1(axis=1) or 1 by D(axis=0) containing the pairwise along spec axis squared Euclidean distances.
        """
        self.check_parwise()     

        return np.sum((self.A-self.B)**2,axis=axis)
    
    def dist_squared_sum(self):
        self.check_parwise()
        return np.sum((self.A-self.B)**2)
    
    def cosine_dist(self,X1,X2):
        
        norm1 = np.sum(X1**2,axis=1)[:,None]    
        norm2 = np.sum(X2**2,axis=1)[None]
        norm1[norm1 == 0.0] = 1.0
        norm2[norm2 == 0.0] = 1.0
        dist2 = np.dot(X1,X2.T)**2/norm1/norm2
        
        dist2 = 1 - dist2
        check_array(dist2)
        return dist2
    
    
#Return the length of row with non-null elements
def len_notnull(I):
    return I[0][pd.notnull(I[0])].shape[0]

#Return dynamics data: dr, dtan of the car: num=num (default is agent:0)
def dynamics_data(I,t,num=0):
    ind_x = 2*num
    ind_y = 2*num+1
    if t >0:
        dx = I[t,ind_x]-I[t-1,ind_x]
        dy = I[t,ind_y]-I[t-1,ind_y]
        dr = np.sqrt(dx**2 + dy**2)
        if t >1:
            dx0 = I[t-1,ind_x]-I[t-2,ind_x]
            dy0 = I[t-1,ind_y]-I[t-2,ind_y]
            
            if dx ==0.0:
                tan = 999
            else:
                tan = dy/dx
            if dx0 ==0.0:
                tan0 =999
            else:
                tan0 = dy0/dx0
            dtan = tan - tan0
        else:
            dtan =0.0
    else:
        dr = 0.0
        dtan =0.0
    return dr,dtan

#return dynamics data over all timesteps
#if truncate = True, all data are set starting from timestep=2 (time=-800)
def dynamics_time_data(I,ind_tar,truncate=True,min_t=0,max_t=9):
    ind_x = 2*ind_tar
    xy_t = I[:,ind_x:ind_x+2]
    dxy_t = xy_t[1:]- xy_t[:-1]
    ddxy_t = dxy_t[1:]- dxy_t[:-1]
    r_t = - np.linalg.norm(xy_t,axis=1)   
    dr_t = r_t[1:]- r_t[:-1]
    ddr_t =  dr_t[1:]- dr_t[:-1]

    if truncate:
        res_xy_t,res_dxy_t,res_ddxy_t,res_r_t,res_dr_t,res_ddr_t = xy_t[2:], dxy_t[1:], ddxy_t, r_t[2:], dr_t[1:], ddr_t
        
        res_xy_t,res_dxy_t,res_ddxy_t,res_r_t,res_dr_t,res_ddr_t = res_xy_t[min_t:max_t],res_dxy_t[min_t:max_t],res_ddxy_t[min_t:max_t],res_r_t[min_t:max_t],res_dr_t[min_t:max_t],res_ddr_t[min_t:max_t]
    
    return res_xy_t,res_dxy_t,res_ddxy_t,res_r_t,res_dr_t,res_ddr_t
 
def dynamics_time_data_t0(I,ind_tar,truncate=True,t=0):
    ind_x = 2*ind_tar
    xy_t = I[:,ind_x:ind_x+2]
    dxy_t = xy_t[1:]- xy_t[:-1]
    ddxy_t = dxy_t[1:]- dxy_t[:-1]
    r_t = - np.linalg.norm(xy_t,axis=1)   
    dr_t = r_t[1:]- r_t[:-1]
    #dr_t = np.linalg.norm(dxy_t,axis=1)
    ddr_t =  dr_t[1:]- dr_t[:-1]
    # ddr_t = np.linalg.norm(ddxy_t,axis=1)
    if truncate:
        return xy_t[2:][t], dxy_t[1:][t], ddxy_t[t], r_t[2:][t], dr_t[1:][t], ddr_t[t]

def dynamics_tavg_data(I,ind_tar):
    xy_t,dxy_t,ddxy_t,r_t,dr_t,ddr_t =dynamics_time_data(I,ind_tar,truncate=True,min_t=0,max_t=9)
    return xy_t.mean(axis=0),dxy_t.mean(axis=0),ddxy_t.mean(axis=0),r_t.mean(),dr_t.mean(),ddr_t.mean()


def range_t(a):
    return np.arange(-100*(len(a)-1),100,100)

# def range_plot_tx(x):
#     le=len(x)
#     tx = np.zeros((le,2))
#     tx[:,0] =np.arange(-100*(le-1),100,100)
#     tx[:,1] =x
#     return tx.T



    

    

def Coor_T_xy_array(xy0,xy1,dxy0,dxy1):
    a = dxy0[:,1]/dxy0[:,0]
    c = dxy1[:,0]/dxy1[:,1]
    a = normalize_slope(a)
    c = normalize_slope(c)
    b = xy0[:,1] - a*xy0[:,0]
    d = xy1[:,0] - c*xy1[:,1]
    x_T = (c*b+d)/(1-a*c)
    x_T = normalize_slope(x_T)
    y_T = (a*d+b)/(1-a*c)
    y_T = normalize_slope(y_T)
    return x_T,y_T


def Coor_T_xy(xy0,xy1,dxy0,dxy1):
    a = dxy0[1]/dxy0[0]
    c = dxy1[0]/dxy1[1]
    a = normalize_slope_sc(a)
    c = normalize_slope_sc(c)
    b = xy0[1] - a*xy0[0]
    d = xy1[0] - c*xy1[1]
    x_T = (c*b+d)/(1-a*c)
    x_T = normalize_slope_sc(x_T)
    y_T = (a*d+b)/(1-a*c)
    y_T = normalize_slope_sc(y_T)
    return x_T,y_T
    
def normalize_slope(array):
    array[np.isnan(array)]=999
    return array

def normalize_slope_sc(scaler):
    
    if np.isnan(scaler):
        return 999
    else:
        return scaler
    

    
def I_inds(I,inds,ind_age=0,include_age=False):
    ind_x=inds*2
    ind_y=inds*2+1
    ind_xy=np.concatenate((ind_x[:,None],ind_y[:,None]),axis=1).flatten()
    ind = ind_xy
    if include_age:
        ind_xy_age=np.array([ind_age*2,ind_age*2+1])
        ind = np.concatenate((ind_xy_age,ind_xy))
    return I[:,ind]    
    
def get_rmse(A,B,type_obj='traj'):
    if A.ndim !=2:
        print("Error: input is not matrix.")
    if A.shape != B.shape:
        print("Error: shape not consistent: A(%s),B(%s)."%(A.shape,B.shape))
    if type_obj=='traj':
        n,d=A.shape
        rmse =np.sqrt(np.sum((A-B)**2)/n/d)
    return rmse
    
    
def sign_able(x):
    y=np.sign(x)
    return -(y-1)/2    
    
    
 
def cv_gen_inds(len_dataset,fold):

    return np.arange(0,len_dataset,np.floor(len_dataset/fold),dtype=np.int32)
    
    
def cv_gen_sets(X_train,y_train,fold):
    inds_fold = cv_gen_inds(len(X_train),fold)
    X_cv_train=[]
    X_cv_val=[]
    y_cv_train=[]
    y_cv_val=[]
    for i in range(1,fold+1):
        X_cv_val.append(X_train[inds_fold[i-1]:inds_fold[i]])
        y_cv_val.append(y_train[inds_fold[i-1]:inds_fold[i]])

        if i ==0:
            X_cv_train.append(X_train[inds_fold[i]:])
            y_cv_train.append(y_train[inds_fold[i]:])
        elif i ==fold:
            X_cv_train.append(X_train[:inds_fold[i-1]])
            y_cv_train.append(y_train[:inds_fold[i-1]])
        else:
            X_cv_train.append(X_train[:inds_fold[i-1]]+X_train[inds_fold[i]:])
            y_cv_train.append(y_train[:inds_fold[i-1]]+y_train[inds_fold[i]:])
            
    return X_cv_train,y_cv_train,X_cv_val,y_cv_val
    
    
    
    
    
    
    
 
    
 