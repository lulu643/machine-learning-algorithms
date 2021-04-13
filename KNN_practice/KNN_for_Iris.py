#!/usr/bin/env python
# coding: utf-8

# # KNN for Iris flowers classification

# In[1]:


# import modules
import matplotlib.pyplot as plt
import numpy as np 
from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets as ds
from sklearn.model_selection import train_test_split


# In[2]:


# load iris dataset
iris = ds.load_iris()


# In[3]:


# assign iris features to X, an array of shape (150,4)
# assign iris labels to y, an array of shape (150,)
X = iris['data']
y = iris['target']


# ## (a) calculate elements in each class

# In[4]:


# calculate elements in each class
# print out the result
## to do
classes = {}
for element in y:
    classes[element] = classes.get(element, 0) + 1
for key, value in classes.items():
    print(f'Number of elements in Class{key}: {value}')


# ## (b) build a KNeighborsClassifier with k=1

# In[5]:


# initialize the knn model
model_knn = KNeighborsClassifier(n_neighbors=1)
model_knn.fit(X,y)


# In[6]:


# calculate prediction accuracy
# print out the accuracy
## to do
pred = model_knn.predict(X)
count = 0
for i in range(len(y)):
    if pred[i] == y[i]:
        count += 1
print('The accuracy equals {:.2f}'.format(count/len(y)))
# This is meaningless because k equals 1. The nearest neighbor
# of a point is itself.


# ## (c) find optimal value of k

# In[7]:


# split the dataset into training and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, shuffle=True, random_state=0)


# In[8]:


# try different value of k from 1 to 50
K = 50
train_accu = np.zeros(50)
test_accu = np.zeros(50)
for i in range(1,K+1):
    # initialize the model
    # fit the data
    # store training accuracy in train_accu
    # store validation accuracy in test_acc
    ## to do
    model_knn = KNeighborsClassifier(n_neighbors=i)
    model_knn.fit(X_train,y_train)
    
    # test on training set
    pred = model_knn.predict(X_train)
    count = 0
    for j in range(len(y_train)):
        if pred[j] == y_train[j]:
            count += 1
    train_accu[i-1] = round(count/len(y_train), 2)
    
    # test on testing set
    pred = model_knn.predict(X_test)
    count = 0
    for j in range(len(y_test)):
        if pred[j] == y_test[j]:
            count += 1
    test_accu[i-1] = round(count/len(y_test), 2)


# In[9]:


# plot the training accuracy and test accuracy against k
plt.figure()
plt.xlabel('k')
plt.ylabel('Accuracy (%)')
x_range = np.linspace(1, K, num=K)
plt.plot(x_range, train_accu, label='training')
plt.plot(x_range, test_accu, label='test')
plt.legend()


# In[10]:


# find the optimal k value
# print out the optimal k
## to do
print('The optimal k for this knn model is 10.')


# ## (d) predict a new sample

# In[11]:


# check the order of the features
iris['feature_names']


# In[12]:


# match the input values with the feature names
## to do
new_point = [3.8, 5.0, 1.2, 4.1]


# In[13]:


# make prediction
# print out the prediction result
## to do
model_knn = KNeighborsClassifier(n_neighbors=10)
model_knn.fit(X_train,y_train)
pred = model_knn.predict([new_point])
print('The predicted class is:', pred[0])


# In[ ]:




