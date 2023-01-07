import utils
import numpy as np
import pandas as pd
from scipy import stats


def get_similarity(coor_A,coor_B):
    """
    #find shortest possible coordinate vector min_coor_B in terms of coor_A and return the Distance
    #where min_coor_B is reorganized according to one of the permutaion of [1,2,3,...k] and coor_B
    #eg. ind_B= [3,5,6,1,2,7,...] (10 by 1) use this indice sequence to reoganize coor_B so that coor_B = [x3,y3,x5,y5,x6,y6,x1,y1,...] (20 by 1)
    #find the sequence that returns shortest distance score of Distance Matrix of A,B
    
    #coor_A is a 20 by 1 vector, containing coordinates of car 0-9 from test set
    #coor_B is a 20 by 1 vector, containing coordinates of car 0-9 from training set
    #min_ind_B is a 10 by 1 vector, containing indices of indices of reorganized B, providing shortest distance to A
    #min_coor_B is a 20 by 1 vector, containing coordinates of reorganized B, providing shortest distance to A
    #Dist is the global shortest distance among all possible pair
    #Dist is calculated by summing over all elements from a Distance Matrix which is computed by coor_A and min_coor_B
    
    #Procedure: 
    #1. find next permutation of B
    #2. get Distance_Matrix_temp
    #3. get Dist_temp
    #4. compare Dist_temp vs Dist_min
    #5. update Dist_min #, min_ind_B, min_coor_B
    #6. if this is not the last permutation case, go to 1
    Dist = 0
    raise NotImplementedError # TODO
    return Dist #optional: also return min_ind_B, min_coor_B 
    
    #note that Cost = O(k!*k(k-1)/2)
    #where k is half the lenth of coor_A or coor_B
    #so Cost ~ 163,296,000 times Cost[dist((x0,y0),(x1,y1))]
    """
    



def knn_frame(I,t,k):
    '''Find k nearest neighbor for the agent car, in frame of timestep = f(t) = -1000+100*t, t in (0,...10)    
    
    Input: dataframe of a trajectory file with certain time index t =num_rows
    
    Return: the indices of knn and their distances towards the agent
            ind_knn, dist2_knn are k by 1, 1D arrays
    '''    
    le = utils.len_notnull(I)
    frame_coor = I[t][:le]
    pairwise_coor = np.concatenate((frame_coor[::2][:,None],frame_coor[1::2][:,None]),axis=1)
    coor_tar = pairwise_coor[0][None]
    dist2 = utils.DistanceMatrix(pairwise_coor[1:],coor_tar).euclidean_dist_squared()  
    ind_knn = np.argsort(dist2,axis=0)[:k]
    dist2_knn = np.zeros(k)
    dist2_knn = dist2[ind_knn]
    
    return ind_knn.flatten()+1, dist2_knn.flatten()



# def sign_inter_able(dot_age,dot_oth,t_age,t_oth,r_age_T,r_oth_T,strict_lv='strict_timing',time_cutoff=10,r_T_cutoff=50):

#     return sign_final








class KNN:

    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        self.X= X # just memorize the trianing data
        self.y = y
        self.k = min(self.k, self.X.shape[0])
    
    def predict(self, X_test,method='weight'):
        # Compute distance distances between X and Xtest
        
        
        y_pred = np.zeros((X_test.shape[0],self.y.shape[1]))

        self.find_inds_knn(self.X,X_test)
        if method == 'mean':
            y_pred=np.mean(self.y[self.inds_knn],axis=1)
        # print(self.y.shape)
        # print(y_pred.shape)
        # print(self.inds_knn.shape)
        # print(self.dist2_knn.shape)
        # print(self.y[self.inds_knn].shape)
        if method == 'weight':
            weights=self.nomalize_dist(self.dist2_knn)
            y_weighted=weights[:,:,None]*self.y[self.inds_knn]
            y_pred=np.sum(y_weighted,axis=1)
            
        return y_pred
    
    def find_inds_knn(self,X, X_test):
        dist2 = utils.DistanceMatrix(X,X_test).euclidean_dist_squared()
        self.inds_knn = np.argsort(dist2,axis=0)[:self.k].T
        self.dist2_knn = dist2[self.inds_knn][0].T #???
        #print(self.dist2_knn)
        return self.inds_knn,self.dist2_knn

    def nomalize_dist(self, dist2):
        sum_dist2=np.sum(dist2,axis=1)
        res= dist2/sum_dist2[:,None]
        return res








