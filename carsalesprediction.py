# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 17:59:00 2020

In this project, using the informations of age,gender,annual salary,credit card depth
and net worth,predict how much money he/she will spent to buy a car.

artificialneuralnetwork car sales prediction
@author: Beytu
"""

#1.import libraries
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns


#2.import dataset
car_df=pd.read_csv('Car_Purchasing_Data.csv',encoding='ISO-8859-1')
ch=car_df.head()
cht=car_df.tail()

#3.visualize dataset
sns.pairplot(car_df)
plt.show()

#4.create testing and training dataset also data cleaning
X=car_df.drop(['Customer Name','Customer e-mail','Country','Car Purchase Amount'],axis=1)
xh=X.head()
xt=X.tail()
y=car_df['Car Purchase Amount']
xs=X.shape
ys=y.shape

from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
X_scaled=scaler.fit_transform(X)
maxdata=scaler.data_max_
mindata=scaler.data_min_
y=y.values.reshape(-1,1)
y_scaled=scaler.fit_transform(y)

#5.training the model
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_scaled,y_scaled,test_size=0.25,random_state=0)

import tensorflow.keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler


model=Sequential()
model.add(Dense(25,input_dim=5,activation='relu'))
model.add(Dense(25,activation='relu'))
model.add(Dense(1,activation='linear'))


ms=model.summary()
model.compile(optimizer='adam',loss='mean_squared_error')
epochs_hist=model.fit(X_train,y_train,epochs=20,batch_size=25,verbose=1,validation_split=0.2)

#6.evaluating the model
ehh=epochs_hist.history.keys()
plt.plot(epochs_hist.history['loss'])
plt.plot(epochs_hist.history['val_loss'])
plt.title('model progress during training/validation')
plt.xlabel('epoch number')
plt.ylabel('training and validation loss')
plt.legend(['training loss','validation loss'])
plt.show()


#create new test data and predict 
#gender,age,annual salary,credit card debt,net worth
X_testnew=np.array([[1, 50, 50000, 10985, 629312]])
y_predictnew=model.predict(X_testnew)
print('expected purchase amount',y_predictnew[:,0])




