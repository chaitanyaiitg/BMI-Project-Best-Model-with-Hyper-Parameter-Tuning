#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
df=pd.read_csv('F:/Data_Science/Height_Weight_Index.csv')


# In[2]:


df


# In[3]:


df.info()


# In[4]:


df.isnull().sum()


# In[5]:


from sklearn.preprocessing import LabelEncoder
gender=LabelEncoder()
df['Sex']=gender.fit_transform(df['Gender'])


# In[6]:


df['Index'].unique()


# In[7]:


bins=(-1,0,1,2,3,4,5)
health=('Extreme Weak','Weak','Normal','OverWeight','Obesity','Extreme Obesity')
df['Health']=pd.cut(df['Index'],bins=bins, labels=health)


# In[9]:


df


# In[11]:


from sklearn import svm
from sklearn.metrics import confusion_matrix, classification_report,accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import seaborn as sns


# In[12]:


sns.countplot(df['Gender'])


# In[13]:


sns.countplot(x='Health', data=df,palette="Set1")
plt.xticks(rotation=90)


# In[19]:


sns.countplot(x='Health', hue='Gender', data=df,palette="Set2")
plt.xticks(rotation=90)


# In[20]:


sns.relplot(x='Weight',y='Height',hue='Health',data=df )


# In[21]:


sns.relplot(x='Height',y='Weight',hue='Health',data=df,palette="Set1" )


# In[22]:


sns.relplot(x='Health',y='Weight',hue='Gender',kind='line',data=df,palette="Set1",aspect=2.5 )


# In[23]:


sns.relplot(x='Health',y='Height',hue='Gender',kind='line',data=df,palette="Set1" , height=5, aspect=2.5)


# In[24]:


df


# In[25]:


df.drop(['Gender','Index'],axis=1,inplace=True)


# In[27]:


X=df.drop('Health',axis=1)
y=df['Health']


# In[28]:


X


# In[30]:


y


# In[31]:


from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import Lasso
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler


# In[32]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=10)


# In[33]:


s=StandardScaler()
X_train=s.fit_transform(X_train)
X_test=s.transform(X_test)


# In[34]:


svmodel=SVC()
svmodel.fit(X_train,y_train)
svmodel.score(X_test,y_test)


# In[35]:


from sklearn.model_selection import ShuffleSplit
cv = ShuffleSplit(n_splits=5, test_size=0.25, random_state=10)
cross_val_score(SVC(),X,y,cv=cv).mean()


# In[36]:


def cross_val_score_model(model):
    cv = ShuffleSplit(n_splits=5, test_size=0.25, random_state=10)
    cv_score=cross_val_score(model,X,y,cv=cv).mean()
    print('CV_Score' + ' '+ str(model) +': '+ str(cv_score))

cross_val_score_model(SVC())
cross_val_score_model(RandomForestClassifier())
cross_val_score_model(DecisionTreeClassifier())
cross_val_score_model(LogisticRegression(solver='liblinear',multi_class='auto'))


# In[37]:


model_params={
    'svm':{
        'model':SVC(),
        'params':{
            'C':[1,10,100],
            'kernel':['rbf','linear']
        }
    },
    'random_forest':{
        'model':RandomForestClassifier(),
        'params':{
            'n_estimators':[1,5,10,1000],
            'criterion' : ['gini', 'entropy']
        }
    },
    'decisionTree':{
        'model':DecisionTreeClassifier(),
        'params':{
            'criterion' : ['gini', 'entropy']
            }
    },
    'logistic regression':{
        'model':LogisticRegression(solver='liblinear',multi_class='auto'),
        'params':{
            'C':[1,5,10]
        }
            
    }

}


# In[38]:


scores1=[]
cv = ShuffleSplit(n_splits=5, test_size=0.25, random_state=10)
for model_name, mp in model_params.items():
    random_clf=RandomizedSearchCV(mp['model'],mp['params'],cv=cv, return_train_score=False)
    random_clf.fit(X,y)
    scores1.append({
        'model':model_name,
        'best_score':random_clf.best_score_,
        'best param':random_clf.best_params_,
        'best estimator':random_clf.best_estimator_
    })

ds=pd.DataFrame(scores1,columns=['model','best_score','best param','best estimator'])
ds


# In[39]:


def display_text_max_col_width(df, width):
    with pd.option_context('display.max_colwidth', width):
        print(df)

display_text_max_col_width(ds['best param'], 800)


# In[40]:


cross_val_score_model(SVC(kernel='linear',C=1))
cross_val_score_model(RandomForestClassifier(n_estimators= 1000,criterion='entropy'))
cross_val_score_model(DecisionTreeClassifier(criterion='entropy'))
cross_val_score_model(LogisticRegression(solver='liblinear',multi_class='auto',C=5))


# # BEST MODEL : 'SVM MODEL' WITH HYPER PARAMETER TUNING

# In[41]:


svmodel_best=SVC(kernel='linear',C=1)
svmodel_best.fit(X_train,y_train)
svmodel_best.score(X_test,y_test)


# In[49]:


df.iloc[200]


# In[50]:


a=[[184,57,1]]
a=s.transform(a)
b=svmodel_best.predict(a)
b


# In[ ]:




