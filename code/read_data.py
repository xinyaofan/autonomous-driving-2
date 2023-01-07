import pandas as pd
import numpy as np
import glob,os
import progressbar

def read_data(train_max=2400):

    #path_data = '/Users/jinlinzhu/Documents/CPSC340/Final_Q2/data'
    path_train_X = os.path.join('..','data','train','X')
    # file = glob.glob(os.path.join(path_train_X, "*.csv"))
    # X_train = []
    # 
    # train_t=0
    # #print("Loading X_train...")
    # for f in progressbar.progressbar(file):
    #     X_train.append(pd.read_csv(f))
    #     train_t+=1
    #     if train_t == train_max:
    #         break
    train_t=0
    X_train = []
    for i in progressbar.progressbar(range(2308)):
        filename="X_{number}.csv".format(number=i)
        filepath=os.path.join(path_train_X,filename)
        X_train.append(pd.read_csv(filepath))
        # train_t+=1
        # if train_t == train_max:
        #     break

    path_train_y = os.path.join('..','data','train','y')
    # file = glob.glob(os.path.join(path_train_y, "*.csv"))
    # y_train = []
    # #print("Loading y_train...")
    # for f in progressbar.progressbar(file):
    #     y_train.append(pd.read_csv(f))
    y_train=[]   
    for i in progressbar.progressbar(range(2308)):
        filename="y_{number}.csv".format(number=i)
        filepath=os.path.join(path_train_y,filename)
        y_train.append(pd.read_csv(filepath))
    
    path_val_X = os.path.join('..','data','val','X')
    # file = glob.glob(os.path.join(path_val_X, "*.csv"))
    # X_val = []
    #print("Loading X_val...")
    # for f in progressbar.progressbar(file):
    #     X_val.append(pd.read_csv(f))
    X_val = []
    for i in progressbar.progressbar(range(524)):
        filename="X_{number}.csv".format(number=i)
        filepath=os.path.join(path_val_X,filename)
        X_val.append(pd.read_csv(filepath))

    path_val_y = os.path.join('..','data','val','y')
    # file = glob.glob(os.path.join(path_val_y, "*.csv"))
    # y_val = []
    
    # #print("Loading y_val...")
    # for f in progressbar.progressbar(file):
    #     y_val.append(pd.read_csv(f))
    y_val = []
    for i in progressbar.progressbar(range(524)):
        filename="y_{number}.csv".format(number=i)
        filepath=os.path.join(path_val_y,filename)
        y_val.append(pd.read_csv(filepath))
    
    
    
    path_test_X = os.path.join('..','data','test','X')
    # file = glob.glob(os.path.join(path_test_X, "*.csv"))
    # X_test = []
    
    #print("Loading X_test...")
    # for f in progressbar.progressbar(file):
    #     X_test.append(pd.read_csv(f))
    X_test=[]
    for i in progressbar.progressbar(range(20)):
        filename="X_{number}.csv".format(number=i)
        filepath=os.path.join(path_test_X,filename)
        X_test.append(pd.read_csv(filepath))
        
    return X_train,y_train,X_val,y_val,X_test