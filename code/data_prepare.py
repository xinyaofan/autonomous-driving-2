"""
Created on Sun Dec  6 15:12:07 2020

@author: xinyao fan
"""

import numpy as np
import pandas as pd
import cmath
from auto_regression import AutoReg
from read_data import read_data
#from raw data to the numerical matrix 11*20 each row is a time slice.
#set all non-exist cars coordinates to nan and shift to the right most col.
def data_prepare_X(X_raw):
    values=X_raw.values
    ind_age=int(np.where(values[0]==' agent')[0]) #index of agent
    age_xy=values[:,(ind_age+2):(ind_age+4)]
    
    #get non-exist car indices (shift to role indices)
    ind_nan=np.unique(np.where(values[:,6::6]==0)[1]) 
    ind_nan=(ind_nan+1)*6-4
   
    #index, data of others, remove all non-exist indices
    ind_oth=np.asarray(np.where(values[0]==' others')[0])
    ind_oth=np.delete(ind_oth,np.where(ind_oth==ind_nan))
    
    ind_oth_xy=np.concatenate((ind_oth+2,ind_oth+3)).T.flatten()
    oth_xy = values[:,ind_oth_xy] 
    
    #11*20 array, the first two columns are coordinates of agent car
    #while the left colums are coordiates of others car
    #fill rest of the dataset with NaN
    nan_xy=np.empty((len(X_raw),2*len(ind_nan)))
    nan_xy[:]=np.NaN
    X=np.concatenate((age_xy, oth_xy, nan_xy), axis=1)
    return X.astype(float)

def data_prepare_y(y_raw):
    le = len(y_raw) 
    values=y_raw.values
    y_x = values[:,1]
    y_y = values[:,2]
    if le!= 30:

        AR_model_x = AutoReg(lags=2)
        AR_model_x.fit(y_x)
        y_x_pred=AR_model_x.predict(le,29)
        
        AR_model_y = AutoReg(lags=2)
        AR_model_y.fit(y_y)
        y_y_pred=AR_model_y.predict(le,29)
        
        y_x = np.concatenate((y_x,y_x_pred))
        y_y = np.concatenate((y_y,y_y_pred))
    if len(y_x)!=30:
        print("Error: not 30!")
    y=np.zeros((30,2))
    y[:,0] = y_x
    y[:,1] = y_y
    return y


def data_preprocess(dataset,type_dataset='X'):
    dataset_processed=[]
    
    for i in range(len(dataset)):
        if type_dataset=='X':
            dataset_processed.append(data_prepare_X(dataset[i]))
        elif type_dataset=='y':
            dataset_processed.append(data_prepare_y(dataset[i]))

    return dataset_processed

def data_read_preprocessed():
    X_train,y_train,X_val,y_val,X_test=read_data()
    X_train = data_preprocess(X_train)
    y_train = data_preprocess(y_train,type_dataset='y')
    X_val = data_preprocess(X_val)
    y_val = data_preprocess(y_val,type_dataset='y')
    X_test = data_preprocess(X_test)
    return X_train,y_train,X_val,y_val,X_test


#(x,y) to (rho,theta) vector form 
def polar_transform(x):
    #x=f1[:,0:2]
    polar=[]
    for i in range(x.shape[0]):
        tem=complex(x[i,0],x[i,1])
        cn=cmath.polar(tem)
        polar=np.append(polar,cn)
    polar=np.reshape(polar,(x.shape[0],x.shape[1]))
    return polar
    
#absolute coordinate system: the intersection is origin.
def polar_system(f): #numerical matrix
    result=np.zeros((f.shape[0],f.shape[1]))
    num_cols = f.shape[1]
    for i in np.arange(0,num_cols,2):
            print(f[:,i:(i+2)])
            anded = polar_transform(f[:,i:(i+2)])
            result[:,i:(i+2)]=anded
    return result

#relative coordinate system: the agent car is the origin
def relative_polar_system(f): #numerical matrix
    result=np.zeros((f.shape[0],f.shape[1]))
    num_cols = f.shape[1]
    for i in np.arange(2,num_cols,2):
        tem=f[:,i:(i+2)]-f[:,0:2]
        anded = polar_transform(tem)
        result[:,i:(i+2)]=anded
    return result

    
    
    
    
    