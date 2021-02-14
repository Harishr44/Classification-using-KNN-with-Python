# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 15:11:20 2020

@author: Harish
"""

import pandas as pd
import numpy as np
glass=pd.read_csv("glass.csv")
from sklearn.model_selection import train_test_split
train,test = train_test_split(glass,test_size = 0.2,random_state=0)
train.iloc[:,9]
from sklearn.neighbors import KNeighborsClassifier as KNC

acc = []
for i in range(3,50,2):
    neigh = KNC(n_neighbors=i)
    neigh.fit(train.iloc[:,0:9],train.iloc[:,9])
    train_acc = np.mean(neigh.predict(train.iloc[:,0:9])==train.iloc[:,9])
    test_acc = np.mean(neigh.predict(test.iloc[:,0:9])==test.iloc[:,9])
    acc.append([train_acc,test_acc])



import matplotlib.pyplot as plt # library to do visualizations 

# train accuracy plot 
plt.plot(np.arange(3,50,2),[i[0] for i in acc],"bo-")

# test accuracy plot
plt.plot(np.arange(3,50,2),[i[1] for i in acc],"ro-")

plt.legend(["train","test"])
