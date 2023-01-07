import pandas as pd
import numpy as np
import glob,os


path_data = '/Users/xinyaofan/Desktop/cpsc340-main/'
path_train_X = os.path.join(path_data,'data','train','X')
os.chdir(path_train_X)
#print(path_train_X)
#file = glob.glob(os.path.join(path_train_X, "*.csv"))
#file = glob.glob(os.path.join(path_train_X, "*.csv"))

X_train = []
for i in range(2308):
   filename="X_{number}.csv".format(number=i)
   X_train.append(pd.read_csv(filename))

#print(X_train[59])
#read train y
y_train=[]
path_train_y = os.path.join(path_data,'data','train','y')
os.chdir(path_train_y)
for i in range(2308):
    filename="y_{number}.csv".format(number=i)
    y_train.append(pd.read_csv(filename))

print(y_train[66])

#read validation x
X_val=[]
path_val_X = os.path.join(path_data,'data','val','X')
os.chdir(path_val_X)

for i in range(524):
    filename="X_{number}.csv".format(number=i)
    X_val.append(pd.read_csv(filename))

print(X_val[66])

#read validation y
y_val=[]
path_val_y = os.path.join(path_data,'data','val','y')
os.chdir(path_val_y)
for i in range(524):
    filename="y_{number}.csv".format(number=i)
    y_val.append(pd.read_csv(filename))

print(y_val[33])

#read test X
X_test=[]
path_test_X = os.path.join(path_data,'data','test','X')
os.chdir(path_test_X)
for i in range(20):
    filename="X_{number}.csv".format(number=i)
    X_test.append(pd.read_csv(filename))

print(X_test[4])

