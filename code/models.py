# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 12:16:29 2020

@author: qq429
"""

from utils import sign_able
from utils import get_rmse
from utils import len_notnull
from utils import dynamics_tavg_data
from utils import dynamics_time_data_t0
from utils import Coor_T_xy_array
import numpy as np
from knn import KNN














class CompareTrajAgeKNN():
    def __init__(self,X_train,y_train,k):
        '''
        Parameters
        ----------
        X_train : Dataset type form read_data
        y_train : Dataset type form read_data
        k : int
            number of nearest neighbours (KNN)
        Stores Datasets and initializes submodel with hyperparameter k
        -------
        '''
        self.X_train=X_train
        self.y_train=y_train
        self.k=k
        self.submodel = KNN(k=k)
        


        
    def fit(self,X2test):
        '''
        Parameters
        ----------
        X2test : Dataset type form read_data
            has to be processed using 'data_preapare' and 'subsets2trajs'

        Stores processed trajectory matrix satisfying interface of submodel (KNN)
        Turns in the matrix to be used into submodel (KNN)
        -------
        '''
        self.len_X2test=len(X2test)
        self.feature_selcetion(self.X_train,type_phase='train',method='knn')
        self.X_train_trajs = self.subsets2trajs(self.X_train)
        self.y_train_trajs = self.subsets2trajs(self.y_train,type_dataset='y')
        
        self.feature_selcetion(X2test,type_phase='test',method='knn')
        self.X2test_trajs = self.subsets2trajs(X2test,type_phase='test')
        
        self.submodel.fit(self.X_train_trajs,self.y_train_trajs)
        
    def predict(self):
        '''
        Predict using submodel (KNN) form stored usable matrix
        Returns y_pred with original format: time series
        '''
        y_pred_trajs = self.submodel.predict(self.X2test_trajs)
        #y_pred = self.traj2matrix(y_pred)
        return y_pred_trajs
    
    def val_phase(self,X_val,y_val):
        '''
        Input validation sets, 
        fit and predict using the self model
        returns rmse, flattened y_pred and corresponding y_val
        '''
        self.fit(X_val)
        y_pred_trajs = self.predict()
        y_val_trajs = self.subsets2trajs(y_val,type_dataset='y',type_phase='test')
        rmse = get_rmse(y_pred_trajs,y_val_trajs,type_obj='traj')
        return rmse,y_pred_trajs,y_val_trajs
    
    def feature_selcetion(self,dataset,type_phase='train',method='knn'):
        '''
        method == 'knn':
            Select the features of each traj file, using find_knn_subsets function
            Then all ids of selected dim are stored in lists: inds_knn_subset, inds_subsets_knn_classes
        phase_type=='train':
            Filters and stores ids of features of training set , otherwise of test set
        '''
        self.type_phase=type_phase
        if method=='defualt':

            self.knn_classes=np.array([9],dtype=np.int32)
            
            inds_defult=np.array([0,1,2,3,4,5,6,7,8])
            inds_defult=np.repeat(inds_defult[:,None],len(self.X_train)-1,axis=1).T
            
            #store the indices of its knn regarding the subset index
            self.inds_knn_subset=inds_defult
            
            #store the indices of subsets regarding the knn class number
            self.inds_subsets_knn_classes=[[]]
            self.inds_subsets_knn_classes[0].append(list(range(len(self.X_train))))
        if method=='knn':
            self.knn_classes=[0] #0 means 0 neighbor near agent, 1 means 1 , so on and so forth
            
            if type_phase=='train':
                
                self.inds_subsets_knn_classes,self.inds_knn_subset = self.find_knn_subsets(dataset)
            
            if type_phase=='test':
                self.test_inds_subsets_knn_classes,self.test_inds_knn_subset = self.find_knn_subsets(dataset)
            
    
    def find_knn_subsets(self,dataset):
        '''
        ----------
        Variables:
            
        inds_knn_subset : N by k list, dtype= int, where k is k_knn (0 means only agent, 1 means 1 neighbor)      
            Stores the id of sorted k nearest neighbours (including agent as id=0)
        
        inds_subsets_knn_classes: len(knn_classes) by * list, dtype= int
            Stores the id of subsets with corresponding number of nearest neighbours
            
            eg: 
            for training phase, inds_subsets_knn_classes[1]=[2,3] 
            means in subsets X_train[2], X_train[3] only two cars including agent are considered correlated,
            thus being classified to the id=1 in knn_class=[0,1]
        
        Description:
            using KNN to find the k neareset neighbours of the first row (agent) from the traj matrix within the traj file, 
            where traj matrix is composed by trajs of all cars in a single traj file
            inds filter with more strict rule could be added after the implementation of KNN model
            then ids of k nearest neighbours for each traj file (like X_train[i]) are stored 
            and ids of subsets are classified according to the (filtered) number of knn
        
        Returns inds_subsets_knn_classes, inds_knn_subset
        -------

        '''

        #initialize
        inds_knn_subset=[]
        inds_subsets_knn_classes=[[]]
        
        for i in range(len(dataset)):
            model = KNN(k=max(self.knn_classes)+1)
            subset_trajs_mat=self.subsets2trajs(dataset[i],type_dataset='subset')

            inds_knn,_=model.find_inds_knn(subset_trajs_mat[0:],subset_trajs_mat[0][None])
            inds_knn_list=inds_knn.flatten().tolist()
            # inds_knn_list=[x+1 for x in inds_knn_list]
            # inds_knn_list_age=[0]+inds_knn_list


            #add inds filter here
            #replace next inds_knn_subset[i] by new indices
            #then the filtered k might be smaller than the input k at the init
            #if len(inds_knn.flatten()) <= max(self.knn_classes): #ignore?
            
            inds_knn_subset.append(inds_knn_list)  

                
            inds_subsets_knn_classes[len(inds_knn_list)-1].append(i) ###???????????????????
        return inds_subsets_knn_classes, inds_knn_subset
    
    
    def subset_selection(self,subset,id_sel,axis=0):
        '''        
        Parameters
        ----------
        dataset : list
            Dataset like X_train after read_data()
        
        ind_subset : int
            Index of selected subset from Dataset
        
        id_sel: 1D numpy array of int
            IDs of car selected (or timesteps if axis=1)
        
        axis : 1: select the whole row regarding indices as of rows
               0: select the whole column regarding indices as of columns
            
            DESCRIPTION. The default is 1.

        Returns
        -------
        subset_sel : single subset from Dataset with selected columns of id_sel  

        '''
        
        if axis == 0:#defualt select whole columns
            inds_dim_xy = self.id2xy(id_sel)
            subset_sel=subset[:,inds_dim_xy]
        elif axis == 1:
            inds_dim_xy = self.id2xy(id_sel)
            subset_sel=subset[inds_dim_xy]        
        return subset_sel    
    
        
    def id2xy(self,id_cars):
        '''
        Input: virtual id(not the id of cars) of others or agent(id=0) in order of the prepared subset 

        Returns real indices of xy in subset from Dataset
        -------

        '''
        if type(id_cars)==list:
            id_cars=np.asarray(id_cars,dtype=np.int32)
        ind_x=id_cars*2
        ind_y=id_cars*2+1

        if type(id_cars) is int :
            ind_xy=np.array([0,1])
            ind_xy[0]=ind_x
            ind_xy[1]=ind_y
        else:
            ind_xy=np.concatenate((ind_x[:,None],ind_y[:,None]),axis=1).flatten()
        
        return ind_xy
    
     
    def traj2matrix(self,traj):
        '''
        Input flattened subsets
        Returns original subsets
        '''
        
        raise NotImplementedError # TODO
        
    def subsets2trajs(self,dataset,knn_class=0,type_dataset='X',type_phase='train'):
        '''
        Parameter:  
            
        dataset: list
            list of traj files, such as X_train or y_train 
       
        knn_class: int
            only required for 'X' dataset_type
            knn_class=0 means only agent is considered, 1 means 1 nearset neighbor class, so on and so forth.
        
        type_dataset: string
            marks the type of dataset, X means input matrix, y means output matrix, subset means X[i] which is the traj file
        
        self.type_phase: string
            marks the phase of making the matrix, 'train' means training phase, 'test' means test phase
            'train phase' makes adjacent traj matrix for training set 
            while during 'test phase' makes adjacent traj matrix for test set
        
        Describtion:
            Create the matrix of trajectories as rows, which statisfies the interface of original KNN
            So that for each row represents flattened traf from X_train[i] or y_train[i], with num_cars*2*timesteps being dims
        
            First dim of trajs_mat is the length of X_train
        
        Returns trajs_mat
        --------
        '''
        len_dataset=len(dataset)
        
        
        
        if type_dataset=='subset':

            
            num_cars=int(len_notnull(dataset)/2)
            trajs_mat = np.zeros((num_cars,len_dataset*2)) 
            for i in range(num_cars):
                car_xyt =self.subset_selection(dataset,i)
                
                trajs_mat[i] = car_xyt.flatten()
                                                       
        elif type_dataset=='X' and type_phase=='train':
            inds_subset = self.inds_subsets_knn_classes[knn_class]
            num_subsets=len(inds_subset)
            trajs_mat = np.zeros((num_subsets,len(dataset[0])*(knn_class+1)*2))
            for i in range(num_subsets):
                ind_subset=inds_subset[i]
                
                subset=dataset[ind_subset]
                
                subset_sel = self.subset_selection(subset,self.inds_knn_subset[ind_subset]) #inds_subset includes 0 as agent for each row

                trajs_mat[i] = subset_sel.flatten()
        
        elif type_dataset=='X' and type_phase=='test':
            inds_subset = self.test_inds_subsets_knn_classes[knn_class]
            num_subsets=len(inds_subset)
            trajs_mat = np.zeros((num_subsets,len(dataset[0])*(knn_class+1)*2))
            for i in range(num_subsets):
                ind_subset=inds_subset[i]
                
                subset=dataset[ind_subset]
                
                subset_sel = self.subset_selection(subset,self.test_inds_knn_subset[ind_subset]) #inds_subset includes 0 as agent for each row

                trajs_mat[i] = subset_sel.flatten()
        
        elif type_dataset=='y' and type_phase=='train':
            inds_subset = self.inds_subsets_knn_classes[knn_class]
            num_subsets=len(inds_subset)
            trajs_mat = np.zeros((num_subsets,len(dataset[0])*2))
            for i in range(num_subsets):
                ind_subset=inds_subset[i]
                subset_sel = dataset[ind_subset]

                trajs_mat[i] = subset_sel.flatten()
        
        elif type_dataset=='y' and type_phase=='test':

            trajs_mat = np.zeros((self.len_X2test,len(dataset[0])*2))
            for i in range(self.len_X2test):
                
                subset_sel = dataset[i]

                trajs_mat[i] = subset_sel.flatten()
        
        return trajs_mat
    
    
    
class EncounterabilityKNN(CompareTrajAgeKNN):
    def __init__(self,X_train,y_train,k_list=[6,6]):

        self.X_train=X_train
        self.y_train=y_train
        #k is the k_knn of similar trajfiles regarding the test trajfile
        #type of classes of different k-knn, 1 means agent itself, 2 means 1 NN
        
        
        self.k_list=k_list
        self.submodels=[]
        self.submodels.append(KNN(k=self.k_list[0]))
        self.submodels.append(KNN(k=self.k_list[1]))
    
        
        self.Hparameter_init()
        #initialize submodels (KNN in terms of multiple traj knn_classes)

            
            
    def Hparameter_init(self,filter_lv='strict_time',
                        r_T_cutoff=10,time_cutoff=10,HalfCarLength_PassingTime=0.5,
                        knn_classes=2,k_traj_knn=10):
        self.filter_lv=filter_lv
        
        self.r_T_cutoff=r_T_cutoff
        self.time_cutoff=time_cutoff
        self.HalfCarLength_PassingTime=HalfCarLength_PassingTime
        
        self.k_traj_knn=k_traj_knn
        self.knn_classes=knn_classes 
        
        if knn_classes:
            self.knn_classes=knn_classes
        if k_traj_knn:
            self.k_traj_knn=k_traj_knn
        

        
        
        # self.k_list =np.array([self.k0,self.k1],dtype=np.int8)
        
        # # if not k_list: 
        # #     self.k_list=np.ones(self.knn_classes,dtype=np.int8)*self.k   
        # #     print("K_list arranged as [k,k]")
        
        # if len(self.k_list) != self.knn_classes:
        #     print("Error: k_list input form incorrect!")
        #     return
            
    def fit(self,X2test):
        '''
        Parameters
        ----------
        X2test : Dataset type form read_data
            has to be processed using 'data_preapare' and 'subsets2trajs'

        Stores processed trajectory matrix satisfying interface of submodel (KNN)
        Turns in the matrix to be used into submodel (KNN)
        -------
        '''
        
        self.len_X2test=len(X2test)
        
        self.feature_selcetion(self.X_train,type_phase='train')
        self.feature_selcetion(X2test,type_phase='test')
        self.X_train_trajs = []
        self.y_train_trajs = []
        self.X2test_trajs = []
        
        
        for i in range(self.knn_classes):

            self.X_train_trajs.append(self.subsets2trajs(self.X_train,knn_class=i,type_phase='train'))
            self.y_train_trajs.append(self.subsets2trajs(self.y_train,knn_class=i,type_dataset='y',type_phase='train'))#???
            self.X2test_trajs.append(self.subsets2trajs(X2test,knn_class=i,type_phase='test'))
            
            self.submodels[i].fit(self.X_train_trajs[i],self.y_train_trajs[i])#???
            self.check_sanity(self.X_train_trajs[i],self.inds_subsets_knn_classes[i],i)
            self.check_sanity(self.y_train_trajs[i],self.inds_subsets_knn_classes[i],i)
            self.check_sanity(self.X2test_trajs[i],self.test_inds_subsets_knn_classes[i],i)
    
    def check_sanity(self,A,B,c=1):
        if len(A)!=len(B):
            print("Error: Class %d: lens between A(%d) and B(%d) mismached."%(c,len(A),len(B)))
    
    def predict(self):
        '''
        Predict using submodel (KNN) form stored usable matrix
        Returns y_pred with original format: time series
        '''
        y_pred_trajs=[]
        y_pred_trajs_final = np.zeros((self.len_X2test,self.y_train_trajs[0].shape[1])) 

        for i in range(self.knn_classes):
            y_pred_trajs.append(self.submodels[i].predict(self.X2test_trajs[i]))
            y_pred_trajs_final[self.test_inds_subsets_knn_classes[i]]=y_pred_trajs[i]

        return y_pred_trajs_final

    
    def feature_selcetion(self,dataset,type_phase='train'):
        '''
        method == 'knn':
            Select the features of each traj file, using find_knn_subsets function
            Then all ids of selected dim are stored in lists: inds_knn_subset, inds_subsets_knn_classes
        phase_type=='train':
            Filters and stores ids of features of training set , otherwise of test set
        '''
         #0 means 0 neighbor near agent, 1 means 1 , so on and so forth
        
        if type_phase=='train':
            
            self.inds_subsets_knn_classes,self.inds_knn_subset = self.find_knn_subsets(dataset)
        
        if type_phase=='test':
            self.test_inds_subsets_knn_classes,self.test_inds_knn_subset = self.find_knn_subsets(dataset)
    
    
    
    
    
    def find_knn_subsets(self,dataset):

        #initialize
        inds_knn_subset=[]
        inds_subsets_knn_classes=[]
        for i in range(self.knn_classes):
            inds_subsets_knn_classes.append([])
            
        for i in range(len(dataset)):
            model = KNN(k=self.k_traj_knn)
            subset_trajs_mat=self.subsets2trajs(dataset[i],type_dataset='subset')

            inds_knn,_=model.find_inds_knn(subset_trajs_mat[0:],subset_trajs_mat[0][None])
            inds_knn_flat=inds_knn.flatten()


            #add inds filter here

            inds_knn=self.inds_knn_filter(dataset[i],inds_knn_flat[1:])

            
            #replace next inds_knn_subset[i] by new indices
            #then the filtered k might be smaller than the input k at the init
            inds_knn_list=[0]+inds_knn.tolist()
            
            if len(inds_knn_list) > self.knn_classes:
                inds_knn_list=inds_knn_list[:self.knn_classes]
            
            inds_knn_subset.append(inds_knn_list)    
            inds_subsets_knn_classes[len(inds_knn_list)-1].append(i)
        
        return inds_subsets_knn_classes, inds_knn_subset


    def inds_knn_filter(self,trajfile,ind_knn):
        xy_T,t_age,t_oth,dot_age,dot_oth,r_age_T,r_oth_T = self.Coor_T_knn_tavg(trajfile,ind_knn)
        
        time_cutoff = self.time_cutoff
        r_T_cutoff = self.r_T_cutoff
        sign_dot_age=sign_able(dot_age)
        sign_dot_oth=sign_able(dot_oth)
        if self.filter_lv == 'loose': 
            sign_final = sign_dot_age * sign_dot_oth
        elif self.filter_lv == 'strict_time':
            sign_time_dif = sign_able(t_oth-t_age)
            sign_time_age = sign_able(t_age-time_cutoff)
            sign_time_oth = sign_able(t_oth-time_cutoff)
            sign_final = sign_dot_age * sign_dot_oth * sign_time_dif * sign_time_age * sign_time_oth
        elif self.filter_lv == 'strict_space':
            sign_space_age = sign_able(r_age_T - r_T_cutoff)
            sign_space_oth = sign_able(r_oth_T - r_T_cutoff)
            sign_final = sign_dot_age * sign_dot_oth * sign_space_age * sign_space_oth
        elif self.filter_lv == 'strict_hybrid':
            sign_time_oth = sign_able(t_oth-time_cutoff)
            sign_space_age = sign_able(r_age_T - r_T_cutoff)
            sign_space_oth = sign_able(r_oth_T - r_T_cutoff)
            sign_final = sign_dot_age * sign_dot_oth * sign_space_age * sign_space_oth * sign_time_oth
        if len(sign_final) != len(ind_knn):
            print("Error: size of signs and ind_knn not match!")
        ind_knn_new = ind_knn[sign_final!=0]
        return ind_knn_new
    
    def Coor_T_knn_tavg(self,trajfile,ind_knn):
        ind_age=0
        le = len(ind_knn)
        xy_age, dxy_age, ddxy_age, r_age, dr_age, ddr_age = np.zeros((le,2)),np.zeros((le,2)),np.zeros((le,2)),np.zeros(le),np.zeros(le),np.zeros(le)
        xy_oth, dxy_oth, ddxy_oth, r_oth, dr_oth, ddr_oth = np.zeros((le,2)),np.zeros((le,2)),np.zeros((le,2)),np.zeros(le),np.zeros(le),np.zeros(le)
        xy_age0, dxy_age0, ddxy_age0, r_age0, dr_age0, ddr_age0 = dynamics_time_data_t0(trajfile,ind_age) 
        for ind in range(le):
           xy_age[ind], dxy_age[ind], ddxy_age[ind], r_age[ind], dr_age[ind], ddr_age[ind] = xy_age0, dxy_age0, ddxy_age0, r_age0, dr_age0, ddr_age0
           xy_oth[ind], dxy_oth[ind], ddxy_oth[ind], r_oth[ind], dr_oth[ind], ddr_oth[ind] = dynamics_tavg_data(trajfile,ind_knn[ind]) 
           
        
        x_T,y_T = Coor_T_xy_array(xy_age,xy_oth,dxy_age,dxy_oth)
        xy_T = np.concatenate((x_T[:,None],y_T[:,None]),axis=1)
        
        xy_age_T = xy_age-xy_T
        xy_oth_T = xy_oth-xy_T
        xy_oth_tail_T = xy_oth-xy_T - self.HalfCarLength_PassingTime * dxy_oth
        
        r_age_T = np.linalg.norm(xy_age_T,axis=1)
        r_oth_T = np.linalg.norm(xy_oth_T,axis=1)
        
        t_age = r_age_T/dr_age
        t_oth = r_oth_T/dr_oth
        
        dot_age = np.diag(xy_age_T@dxy_age.T)
        dot_oth = np.diag(xy_oth_tail_T@dxy_oth.T)
        
        return xy_T,t_age,t_oth,dot_age,dot_oth,r_age_T,r_oth_T

#if input <0 output 1; input >0 output 0; input = 0 output 0.5
        
        
        
        
        
        
        
        
        
        
        
        