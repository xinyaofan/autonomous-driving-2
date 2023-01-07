#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import os
import argparse
import utils
import progressbar
import knn
from animation import AnimatedAnalysis
from read_data import read_data


from data_prepare import data_read_preprocessed

from models import CompareTrajAgeKNN
from models import EncounterabilityKNN
import warnings
warnings.filterwarnings("ignore")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-q','--question', required=True)
    parser.add_argument('-k','--k_knn',required=False)
    
    io_args = parser.parse_args()
    question = io_args.question
    io_args = parser.parse_args()
    
    if question == "test":

        test_range=100
        animation_scale=15
        strict_lv='strict_hybrid'
        test_t = 0
        k_test=11
        time_cutoff_test=100 
        space_cutoff_test=animation_scale
        X_train,y_train,X_val,y_val,X_test=data_read_preprocessed()
        
        model=EncounterabilityKNN(X_train,y_train,k=k_test)
        
        model.filter_lv=strict_lv
        model.time_cutoff=time_cutoff_test
        model.r_T_cutoff=space_cutoff_test
        for i in range(test_range):#progressbar.progressbar(range(test_range)):
            I1=X_train[i]
            ind_knn, dist2_knn = knn.knn_frame(I1,0,k_test)
            
            ind_knn_new = model.inds_knn_filter(I1,ind_knn)
            print(ind_knn_new)
            # AnimatedAnalysis(I1,ind_knn_new,num=i,scale=animation_scale)
        
    elif question == "cross_val":
        
        fold =5
        X_train,y_train,X_val,y_val,X_test=data_read_preprocessed()
        X_cv_train,y_cv_train,X_cv_val,y_cv_val=utils.cv_gen_sets(X_train,y_train,fold)
        rmse_min=100
        k_list=[]
        rmse_list=[]
        for k in range(20):
            for i in range(fold):
                model = CompareTrajAgeKNN(X_cv_train[i],y_cv_train[i],k=k)
                rmse_val,_,_= model.val_phase(X_cv_val[i],y_cv_val[i])
                print ('k = %d, rmse = %.3f'%(k,rmse_val))
                if rmse_val < rmse_min:
                    rmse_min=rmse_val
                    k_min=k
        print('k_min= %d, rmse_min = %.6f'%(k_min,rmse_min))
    elif question == "animation":
        X_train,y_train,X_val,y_val,X_test = data_read_preprocessed()
        #print("Creating Animation Files...")
        for i in progressbar.progressbar(range(100)):
            I1=X_train[i]
            utils.AnimatedScatter(I1,num=i,scale=20)
        
    
    elif question =="val_phase":
        if io_args.k_knn:
            k_val = int(io_args.k_knn)
        else:
            k_val = 6
        
        X_train,y_train,X_val,y_val,X_test=data_read_preprocessed()
        
        model=EncounterabilityKNN(X_train,y_train,k=k_val)
    
        rmse_val,y_pred_trajs,y_val_trajs= model.val_phase(X_val,y_val)
        print ('k = %d, rmse = %.6f'%(k_val,rmse_val))
        print(y_pred_trajs[0,:10])
        print(y_val_trajs[0,:10])
    
    
    elif question =="test_phase":
        if io_args.k_knn:
            k_val = int(io_args.k_knn)
        else:
            k_val = 6
        X_train,y_train,X_val,y_val,X_test=data_read_preprocessed()
        # model=EncounterabilityKNN(X_train,y_train,k=k_val)
        k_list=[7,5]
        model=EncounterabilityKNN(X_train,y_train,k_list=k_list)
        model.Hparameter_init(filter_lv='strict_hybrid',r_T_cutoff=20,time_cutoff=50,HalfCarLength_PassingTime=0.6)

        model.fit(X_test)
        
        y_pred=model.predict()
        print(y_pred.flatten().shape)
        print(y_pred.flatten())
        d = {'location':y_pred.flatten()}
        df = pd.DataFrame(data=d)
        df.to_csv(os.path.join('..','data','test_result.csv'))
        
    elif question =="val_tune":
        X_train,y_train,X_val,y_val,X_test = data_read_preprocessed()
        if io_args.k_knn:
            k_val_max = int(io_args.k_knn)
        else:
            k_val_max = 10
        
        
        
        
        rmse_min=100
        #rmse=[]
        for i in range(1,k_val_max):
            for j in range(1,k_val_max):
                k_list=[i,j]
                model=EncounterabilityKNN(X_train,y_train,k_list=k_list)
                model.Hparameter_init(filter_lv='strict_hybrid',r_T_cutoff=20,time_cutoff=50,HalfCarLength_PassingTime=0.6)
                rmse_val,_,_ = model.val_phase(X_val,y_val)
                #rmse.append(rmse_val)
                if rmse_val<rmse_min:
                    rmse_min=rmse_val
                    k_list_min_rmse = k_list
                print ('k_list = %s, rmse = %.3f'%(k_list,rmse_val))
        
        print("Knn classes: training set:")
        print(len(model.inds_subsets_knn_classes[0]),len(model.inds_subsets_knn_classes[1]))
        print("Knn classes: test set:")
        print(len(model.test_inds_subsets_knn_classes[0]),len(model.test_inds_subsets_knn_classes[1]))
        print('k_list_min_rmse= %s, rmse_min = %.6f'%(k_list_min_rmse,rmse_min))
        
        
    