# -*- coding: utf-8 -*-
import utils
from knn import knn_frame
from knn import ind_knn_filter

class Intersection_model(object):
    def __init__(self,X_train,y_train,X_val,y_val,X_test):
        raise NotImplementedError # TODO







class Hyper_scan(object):
    def __init__(self,X_train,y_train,X_val,y_val,X_test):
        raise NotImplementedError # TODO
    
    def Hparameter_init(self,max_num_knn,space_cutoff,time_cutoff):
        raise NotImplementedError # TODO
    
    
    def tavg_data_knn(self):
        raise NotImplementedError # TODO
    
    
    def interested_data_iter(self,I_iter):
        I = I_iter
        frame_t = self.frame_t
        max_num_knn = self.max_num_knn
        
        tavg_drr_age = utils.dynamics_time_data(I,0)[-1].mean()

        ind_knn = knn_frame(I,frame_t,max_num_knn)
        ind_knn_new = ind_knn_filter(I,ind_knn)
        num_knn_new = len(ind_knn_new)
        
        return num_knn_new, tavg_drr_age
    
    def interested_data(self):
        self.data_keywords = ['num_knn_new', 'tavg_drr_age']
        return self.data_keywords
    
    def stat_data(self):
        #do: cal mean value of num_knn_new over all dataset
        #ob: mean change vs Hpar: strictions parameters change
        raise NotImplementedError # TODO
    
    def hist_data(self):
        #do: count frequence of various num_knn_new through all dataset, 
        #ob: hist change vs Hpar: strictions parameters change
        raise NotImplementedError # TODO
        
        
        
        
