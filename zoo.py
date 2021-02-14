# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 14:47:59 2020

@author: Harish
"""

import pandas as pd
import numpy as np
zoo=pd.read_csv("Zoo.csv")

from sklearn.model_selection import train_test_split
train,test = train_test_split(zoo,test_size = 0.2,random_state=0)

from sklearn.neighbors import KNeighborsClassifier as KNC

neigh = KNC(n_neighbors= 3)

# Fitting with training data 
neigh.fit(train.iloc[:,1:18],train.iloc[:,0])
pred_train=neigh.predict(train.iloc[:,1:18])
c=pd.crosstab(pred_train,train.iloc[:,0])
train_acc = np.mean(neigh.predict(train.iloc[:,1:18])==train.iloc[:,0])
#27.5
test_acc = np.mean(neigh.predict(test.iloc[:,1:18])==test.iloc[:,0])
#4.76
from sklearn.neighbors import KNeighborsClassifier as KNC

acc = []
for i in range(3,50,2):
    neigh = KNC(n_neighbors=i)
    neigh.fit(train.iloc[:,1:18],train.iloc[:,0])
    train_acc = np.mean(neigh.predict(train.iloc[:,1:18])==train.iloc[:,0])
    test_acc = np.mean(neigh.predict(test.iloc[:,1:18])==test.iloc[:,0])
    acc.append([train_acc,test_acc])



import matplotlib.pyplot as plt # library to do visualizations 

# train accuracy plot 
plt.plot(np.arange(3,50,2),[i[0] for i in acc],"bo-")

# test accuracy plot
plt.plot(np.arange(3,50,2),[i[1] for i in acc],"ro-")

plt.legend(["train","test"])