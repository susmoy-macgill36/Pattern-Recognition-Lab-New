#!/usr/bin/env python
# coding: utf-8

# Assignment Name: Designing a minimum distance to class mean classifier
# 
# Submitted By:
# 
# Name: Susmoy Chakraborty
# 
# ID: 15-02-04-114
# 
# Section: B 
# 
# Group: B2 
# 
# Submission Date: 30-7-2019
# 
# 
# IDE: Jupyter notebook 
# 

# Q: Plot all sample points (train data) from both classes, but samples from the same class should
# have the same color and marker.

# In[1]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


# In[2]:


df = pd.read_csv('train.txt', delimiter =" ",names =['x','y', 'classname'],header=None)


# In[3]:


print(df)


# In[4]:


x = df['x'].tolist()
y= df['y'].tolist()
classes =df['classname'].tolist()


# In[5]:


class1Indexes =[]
class2Indexes =[]


# In[6]:


for i in range(len(classes)):
     if classes[i]==1:
        class1Indexes.append(i)
     else:
        class2Indexes.append(i)
    


# In[7]:


x_class1 =[]
x_class2 =[] 
y_class1 =[]
y_class2 =[] 


# In[8]:


for i in range(len(class1Indexes)):
    x_class1.append(x[class1Indexes[i]])
    

    
for i1 in range(len(class2Indexes)):
    x_class2.append(x[class2Indexes[i1]])
    


# In[9]:


for i2 in range(len(class1Indexes)):
    y_class1.append(y[class1Indexes[i2]])
    

    
for i3 in range(len(class2Indexes)):
    y_class2.append(y[class2Indexes[i3]])
    


# In[96]:


def plotMain():
     plt.scatter(x_class1,y_class1, color='r',marker='o',label='class 1')
     plt.scatter(x_class2,y_class2, color='g',marker='+',label='class 2')
     #plt.scatter(0,2, color='b')
     
        
        
plotMain()
plt.legend()
plt.show()


# Q: Using a minimum distance classifier with respect to ‘class mean’, classify the test data points
# by plotting them with the designated class-color but different marker. Use the given Linear
# Discriminant Function.

# In[11]:


x_classnew =[]
y_classnew =[]


for i in range(len(x_class1)):
    x_classnew.append(x_class1[i])
    x_classnew.append(y_class1[i])
   

for i1 in range(len(x_class2)):
    y_classnew.append(x_class2[i1])
    y_classnew.append(y_class2[i1])


# In[12]:


class1= (np.array(x_classnew)).reshape((len(x_classnew))//2,2)


# In[13]:


class2= (np.array(y_classnew)).reshape((len(y_classnew))//2,2)


# In[14]:


class1_Mean =np.mean(class1,axis=0)


# In[15]:


class2_Mean = np.mean(class2, axis=0)


# In[17]:


df_test = pd.read_csv("test.txt", delimiter = " ", names=['x','y','classname'])


# In[18]:


print(df_test)


# In[73]:


u=[]
for i in range(len(df_test)):
     u.append(df_test['x'][i])
     u.append(df_test['y'][i])   


# In[76]:


testdatas= np.array(u).reshape(len(u)//2,2)


# In[191]:


plotMain()
   

m1 =[]


for i in range(len(testdatas)):
    x=testdatas[i,:]
    #print(x)
    #print(x[0])
    #print(x[1])
    g1 =abs(x.dot(class1_Mean.T)-0.5*(class1_Mean.dot(class1_Mean.T)))
    g2 =abs(x.dot(class2_Mean.T)-0.5*(class2_Mean.dot(class2_Mean.T)))
    
    if g1> g2:
        plt.plot(x[0],x[1], color='r',marker='*')
        m1.append(1)
    else:
        plt.plot(x[0],x[1], color='g',marker='*')
        m1.append(2)


     
        
plt.legend()
plt.show()
        
    


# Q: Find accuracy.

# In[144]:


m1_actual= df_test['classname'].tolist() 


# In[145]:


count1=0
for i in range(len(m1)):
    if m1[i]==m1_actual[i]:
        count1=count1+1
        


# In[146]:


print("ACCURACY:", (count1/len(m1))*100 ,"%")


# Q: Draw the decision boundary between the two-classes.
# 
# 

# In[153]:


f=[]
a1 = df['x'].values
a2 = df['y'].values


for i in range(len(a1)):
    f.append(a1[i])
    f.append(a2[i])
    
a3 = df_test['x'].values
a4 = df_test['y'].values 

for i1 in range(len(a3)):
    f.append(a3[i1])
    f.append(a4[i1])


# In[156]:


max_f=max(f)
min_f=min(f)


# In[183]:


range1 = np.linspace(min_f,max_f,1000)


# In[184]:


shaped_range= range1.reshape(len(range1)//2,2)


# In[188]:


c9=0
for i in range(len(shaped_range)):
    x1=shaped_range[i,:]
    #print(x)
    #print(x[0])
    #print(x[1])
    g1 =x1.dot(class1_Mean.T)-0.5*(class1_Mean.dot(class1_Mean.T))
    g2 =x1.dot(class2_Mean.T)-0.5*(class2_Mean.dot(class2_Mean.T))
    if (g1-g2==0):
        c9=c9+1


# Here, c9 means how for which points g1 = g2 or g1-g2 =0, but c9=0.

# In[ ]:




