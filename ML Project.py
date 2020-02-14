#!/usr/bin/env python
# coding: utf-8

#  DATA EXTRACTION

# In[1]:


import pandas as pd
columns=["age","sex","cp","trestbps","chol","fbs","restecg","thalach","exang","oldpeak","slope","ca","thal","num"]
dataframe=pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data",names=columns)
dataframe


# In[67]:


#EDA
import seaborn as sns
for i in range(0,len(dataframe.columns),5):
    sns.pairplot(dataframe,y_vars=['num'],x_vars=dataframe.columns[i:i+5])


# In[3]:


# Missing data
dataframe=dataframe.replace('?','NaN')
dataframe


# In[69]:


dataframe['ca']=dataframe['ca'].astype(float)
dataframe['thal']=dataframe['thal'].astype(float)


# In[5]:


#Updating missing data with mean
dataframe=dataframe.fillna(dataframe.mean())
dataframe


# In[70]:


#Redundancy check
df=dataframe.duplicated( keep='first')


# In[71]:


df


# In[72]:


import matplotlib
get_ipython().run_line_magic('matplotlib', 'inline')


# In[8]:


#Detecting and removing outiers
dataframe.boxplot(column='age')


# In[9]:


dataframe.boxplot(column='sex')


# In[10]:


dataframe.boxplot(column='cp')


# In[11]:


dataframe.drop(index = dataframe['cp'][dataframe['cp'] < 2].index,inplace=True)
dataframe.boxplot(column='cp')


# In[12]:


dataframe.boxplot(column='trestbps')


# In[13]:


dataframe.drop(index = dataframe['trestbps'][dataframe['trestbps'] >170 ].index,inplace=True)
dataframe.boxplot(column='trestbps')


# In[14]:


dataframe.boxplot(column='chol')


# In[15]:


dataframe.drop(index = dataframe['chol'][dataframe['chol'] > 375 ].index,inplace=True)
dataframe.boxplot(column='chol')


# In[16]:


dataframe.boxplot(column='fbs')


# In[17]:


dataframe.drop(index = dataframe['fbs'][dataframe['fbs'] > 1 ].index,inplace=True)
dataframe.boxplot(column='fbs')


# In[18]:


dataframe.boxplot(column='restecg')


# In[19]:


dataframe.boxplot(column='thalach')


# In[20]:


dataframe.drop(index = dataframe['thalach'][dataframe['thalach'] <90 ].index,inplace=True)
dataframe.boxplot(column='thalach')


# In[21]:


dataframe.boxplot(column='exang')


# In[22]:



dataframe.boxplot(column='oldpeak')


# In[23]:


dataframe.drop(index = dataframe['oldpeak'][dataframe['oldpeak'] > 4 ].index,inplace=True)
dataframe.boxplot(column='oldpeak')


# In[24]:


dataframe.boxplot(column='slope')


# In[25]:


dataframe.boxplot(column='ca')


# In[26]:


dataframe.drop(index = dataframe['ca'][dataframe['ca'] > 2 ].index,inplace=True)
dataframe.boxplot(column='ca')


# In[27]:


dataframe.boxplot(column='thal')


# In[28]:



dataframe['num'].value_counts()


# In[77]:


# Grouping the num column based on Class
dataframe['num'][dataframe['num'] > 1] = 1


# In[30]:


dataframe['num'].value_counts()


# In[31]:


#Splitting training and test set
X=dataframe.values[:,:12]
Y=dataframe.values[:,13]
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=7)


# In[ ]:


X_train


# In[ ]:


X_test


# In[ ]:


Y_train


# In[ ]:


Y_test


# In[36]:


#classification report
cr = metrics.classification_report(Y_test, Y_predicted)
print(cr)


# In[37]:


from sklearn.model_selection import cross_val_score
accuracies=cross_val_score(estimator=model_rf,X=X_train,y=Y_train,cv=10)
accuracies


# In[38]:


accuracies.mean()


# In[39]:


from sklearn.naive_bayes import GaussianNB
model_NB = GaussianNB()
model_NB.fit(X_train,Y_train) 
y_predict = model_NB.predict(X_test)
metrics.accuracy_score(Y_test, Y_predicted)


# In[40]:


cr = metrics.classification_report(Y_test, Y_predicted)
print(cr)


# In[41]:


from sklearn.model_selection import cross_val_score
accuracies=cross_val_score(estimator=model_NB,X=X_train,y=Y_train,cv=10)
accuracies.mean()


# In[42]:


accuracies


# In[43]:


cm=confusion_matrix(Y_test, Y_predicted)
cm
ax= plt.subplot()
sns.heatmap(cm, annot=True, ax = ax); #annot=True to annotate cells

# labels, title and ticks
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
ax.set_title('Confusion Matrix'); 


# In[53]:


from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve, roc_auc_score
 


# In[76]:


#Logistic Regression
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
model_lr = LogisticRegression()
model_lr.fit(X_train, Y_train)
Y_predicted = model_lr.predict(X_test)
model_score = model_lr.score(X_test, Y_test)


# In[63]:


#Predicting the accuracy
metrics.accuracy_score(Y_test, Y_predicted)


# In[64]:


#Confusion matrix
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(Y_test, Y_predicted)
cm


# In[48]:


cr = metrics.classification_report(Y_test, Y_predicted)
print(cr)


# In[49]:


#Plotting the Confusion matrix 
cm=confusion_matrix(Y_test, Y_predicted)
cm
ax= plt.subplot()
sns.heatmap(cm, annot=True, ax = ax); #annot=True to annotate cells

# labels, title and ticks
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
ax.set_title('Confusion Matrix'); 


# In[75]:



#Cross validation
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import cross_val_score
accuracies=cross_val_score(estimator=model_lr,X=X_train,y=Y_train,cv=10)
accuracies.mean()


# In[66]:


accuracies


# In[ ]:




