#!/usr/bin/env python
# coding: utf-8

# ## Data loading and reading

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


df_train=pd.read_csv('loan_sanction_train.csv')
df_test=pd.read_csv('loan_sanction_test.csv')


# In[3]:


df_train.shape


# In[4]:


df_test.shape


# In[5]:


df_train.head()


# In[6]:


df_test.head()


# In[7]:


df_train.columns


# In[8]:


df_test.columns


# In[9]:


df_train['Gender'].value_counts()


# In[10]:


df_train['Married'].value_counts()


# In[11]:


df_train['Dependents'].value_counts()


# In[12]:


df_train['Education'].value_counts()


# In[13]:


df_train['Self_Employed'].value_counts()


# In[14]:


df_train['ApplicantIncome'].nunique()


# In[15]:


df_train['CoapplicantIncome'].nunique()


# In[16]:


df_train['Loan_Amount_Term'].value_counts()


# In[17]:


df_train['Credit_History'].value_counts()


# In[18]:


df_train['Property_Area'].value_counts().plot(kind='bar')
print(df_train['Property_Area'].value_counts())


# In[19]:


import seaborn as sns
sns.countplot(df_train['Credit_History'], hue=df_train['Loan_Status'])


# In[20]:


sns.countplot(df_train['Dependents'], hue=df_train['Loan_Status'])


# In[21]:


sns.countplot(df_train['Self_Employed'], hue=df_train['Loan_Status'])


# In[22]:


df_train.info()


# In[23]:


df_train.info()


# In[24]:


df_train.isna().sum()


# In[25]:


df_test.isna().sum()


# ## Handling Missing values and changing dtype

# In[26]:


df_train['Gender'].fillna('',inplace=True)
df_train['Married'].fillna('',inplace=True)
df_train['Dependents'].fillna('',inplace=True)
df_train['Self_Employed'].fillna('',inplace=True)


# In[27]:


df_test['Gender'].fillna('',inplace=True)
df_test['Married'].fillna('',inplace=True)
df_test['Dependents'].fillna('',inplace=True)
df_test['Self_Employed'].fillna('',inplace=True)


# In[28]:


df_train.info(), df_test.info()


# In[29]:


from sklearn.impute import SimpleImputer

num_imp=SimpleImputer(strategy='median')
num_imp.fit(df_train[['LoanAmount','Loan_Amount_Term','Credit_History']])  


# In[30]:


df_train[['LoanAmount','Loan_Amount_Term','Credit_History']] = num_imp.transform(df_train[['LoanAmount','Loan_Amount_Term','Credit_History']])  


# In[31]:


num_imp2=SimpleImputer(strategy='median')
num_imp2.fit(df_test[['LoanAmount','Loan_Amount_Term','Credit_History']])  
df_test[['LoanAmount','Loan_Amount_Term','Credit_History']] = num_imp2.transform(df_test[['LoanAmount','Loan_Amount_Term','Credit_History']])  


# In[32]:


df_train['Loan_Status']=df_train['Loan_Status'].apply(lambda x :x.replace('Y','1'))
df_train['Loan_Status']=df_train['Loan_Status'].apply(lambda x :x.replace('N','0'))


# In[33]:


df_train.info()


# In[34]:


from sklearn.preprocessing import OneHotEncoder
enc=OneHotEncoder()


# In[35]:


enc_train=enc.fit_transform(df_train[['Gender', 'Married', 'Dependents', 'Education',
       'Self_Employed','Property_Area']])
enc_cols = enc.get_feature_names()


# In[36]:


df_train1=pd.DataFrame(enc_train.toarray(), columns=enc_cols)
df_train1


# In[37]:


enc_test=enc.fit_transform(df_test[['Gender', 'Married', 'Dependents', 'Education',
       'Self_Employed','Property_Area']])
enc_cols_test = enc.get_feature_names()
df_test1=pd.DataFrame(enc_test.toarray(), columns=enc_cols_test)
df_test1


# In[38]:


df_1=pd.concat([df_train1,df_train[['ApplicantIncome',
                                    'CoapplicantIncome','LoanAmount','Loan_Amount_Term','Credit_History',
                                    'Loan_Status']]],axis=1)


# In[39]:


df_1.head()


# In[40]:


df_2=pd.concat([df_test1,df_test[['ApplicantIncome',
                                        'CoapplicantIncome','LoanAmount','Loan_Amount_Term','Credit_History']]],axis=1)


# In[41]:


df_2.head()


# ## Model Building

# In[42]:


X=df_1.drop(['Loan_Status'],axis=1)
Y=df_1['Loan_Status']


# In[43]:


from sklearn.model_selection import train_test_split
X_train, X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=10)


# In[44]:


X_train.shape


# In[45]:


from sklearn.ensemble import GradientBoostingRegressor
model=GradientBoostingRegressor(n_estimators=1500, max_depth=4, min_samples_leaf=5, min_samples_split=8,
                               learning_rate=0.01,loss='huber', random_state=5)

model.fit(X_train,Y_train)


# In[46]:


model.score(X_train,Y_train)


# In[47]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

params = {'max_depth':[1,2,3],'min_samples_leaf':[1,2,3],'min_samples_split':[3,4]}
model = GridSearchCV(DecisionTreeClassifier(random_state=100),param_grid=params,cv=5, scoring="accuracy")
model.fit(X_train,Y_train)


# In[48]:


model.score(X_train, Y_train)


# In[49]:


from sklearn.model_selection import RandomizedSearchCV
params = {'max_depth':[1,2,3,4,5],'min_samples_leaf':[1,2],'min_samples_split':[3,4,5]}
model = RandomizedSearchCV(RandomForestClassifier(random_state=100),
                               param_distributions=params, cv=4, 
                               random_state=1)
model.fit(X_train,Y_train)


# In[50]:


model.score(X_train,Y_train)


# In[51]:


from sklearn.linear_model import LogisticRegression


# In[52]:


modelLR=LogisticRegression()


# In[53]:


modelLR.fit(X_train,Y_train)


# In[54]:


model.score(X_train,Y_train)


# In[55]:


yp=model.predict(df_2)


# In[ ]:




