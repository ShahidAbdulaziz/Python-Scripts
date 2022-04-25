#!/usr/bin/env python
# coding: utf-8

# # DSCI 503 - Homework 08
# ### Shahid Abdulaziz

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report


# ## Problem 1: Diamonds Dataset

# In[2]:


diamonds = pd.read_csv('Diamonds.txt', sep = '\t')

ln_carat = np.log(diamonds.carat)
ln_price = np.log(diamonds.price)

diamonds['ln_carat'] = np.log(diamonds.carat)
diamonds['ln_price'] = np.log(diamonds.price)

diamonds.head(5)


# In[3]:


X1_num = diamonds[['ln_carat'] ].values
X1_cat = diamonds[['cut','color','clarity']].values
y1 = diamonds.ln_price.values

print('Numerical Feature Array Shape:  ' , X1_num.shape)
print('Categorical Feature Array Shape:', X1_cat.shape)
print('Label Array Shape:              ' ,y1.shape)


# In[4]:


encoder = OneHotEncoder(sparse = False)
encoder.fit(X1_cat)
X1_enc = encoder.transform(X1_cat)

print("Encoded Feature Array Shape:",X1_enc.shape)


# In[5]:


X1 = np.hstack((X1_num,X1_enc))

print('Feature Array Shape:', X1.shape)


# In[6]:


X1_train, X1_hold, y1_train, y1_hold = train_test_split(X1, y1, test_size = 0.20, random_state=1)
X1_valid, X1_test, y1_valid, y1_test = train_test_split(X1_hold, y1_hold, test_size = 0.50, random_state=1)


print('Training Features Shape:', y1_train.shape)
print('Validation Features Shape:', y1_valid.shape)
print('Test Features Shape:', y1_test.shape)


# ### Linear Regression Model with One Feature

# In[7]:


dia_mod_1 = LinearRegression()
xshape = X1_train[:,[0]].reshape(1,-1)

dia_mod_1.fit(X1_train[:,[0]], y1_train)

print('Training r-Squared:  ', round(dia_mod_1.score(X1_train[:,[0]], y1_train),4))
print('Validation r-Squared:', round(dia_mod_1.score(X1_valid[:,[0]], y1_valid),4))


# ###  Linear Regression Model with Several Features

# In[8]:


dia_mod_2 = LinearRegression()
dia_mod_2.fit(X1_train,y1_train)

print('Training r-Squared:  ', round(dia_mod_2.score(X1_train, y1_train),4))
print('Validation r-Squared:', round(dia_mod_2.score(X1_valid, y1_valid),4))


# In[9]:


print('Training r-Squared:  ', round(dia_mod_2.score(X1_test, y1_test),4))


# ## Problem 2: Census Dataset

# In[10]:


census = pd.read_csv('census.txt', sep = '\t')
census.head(10)


# In[11]:


census.shape


# In[12]:


census.salary.value_counts().sort_index()


# ### Prepare the Data

# In[13]:


X2_num = census.loc[:, ['age', 'fnlwgt', 'educ_num', 'capital_gain', 'capital_loss', 'hrs_per_week']].values
X2_cat = census.loc[:, ['workclass', 'education', 'marital_status', 'occupation', 'relationship', 'race', 'sex', 'native_country']].values
y2 = census.loc[:, 'salary'].values

print('Numerical Feature Array Shape:  ', X2_num.shape)
print('Categorical Feature Array Shape:', X2_cat.shape)
print('Label Array Shape:              ', y2.shape)


# In[14]:


encoder = OneHotEncoder(sparse = False)
encoder.fit(X2_cat)
X2_enc = encoder.transform(X2_cat)

print("Encoded Feature Array Shape:",X2_enc.shape)


# In[15]:


X2 = np.hstack((X2_num,X2_enc))

print('Feature Array Shape:', X2.shape)


# In[16]:


X2_train, X2_hold, y2_train, y2_hold = train_test_split(X2, y2, test_size = 0.3, random_state=1, stratify=y2)
X2_valid, X2_test, y2_valid, y2_test = train_test_split(X2_hold, y2_hold, test_size = 0.50, random_state=1, stratify=y2_hold)


print('Training Features Shape:', y2_train.shape)
print('Validation Features Shape:', y2_valid.shape)
print('Test Features Shape:', y2_test.shape)


# ###  Logistic Regression Model

# In[17]:


lr_mod = LogisticRegression(solver='lbfgs', penalty='none', max_iter= 10000000000)
lr_mod.fit(X2_train,y2_train)    

print('Training Accuracy:  ', "%.4f"% lr_mod.score(X2_train, y2_train))
print('Validation Accuracy:', "%.4f"% lr_mod.score(X2_test, y2_test))


# ### Decision Tree Models

# In[18]:


dt_train_acc = []
dt_valid_acc = []
depth_range = range(1,30)


for i in depth_range:
    np.random.seed(1)
    temp_tree = DecisionTreeClassifier(max_depth= i, random_state=1)
    temp_tree.fit(X2_train,y2_train)
    temp_tree.score(X2_test, y2_test)
    dt_train_acc.append(temp_tree.score(X2_train, y2_train))
    dt_valid_acc.append(temp_tree.score(X2_test, y2_test)) 
    
    
    
    
dt_idx = np.argmax(dt_valid_acc)
dt_opt_depth = depth_range[dt_idx]

print('Optimal value for max_depth:          ',  "%.4f"% depth_range[dt_idx])
print('Training Accuracy for Optimal Model:  ',  "%.4f"% dt_train_acc[dt_idx])
print('Validation Accuracy for Optimal Model:',  "%.4f"%dt_valid_acc[dt_idx] )


# In[19]:


plt.plot(depth_range,dt_train_acc, label = 'Training')
plt.plot(depth_range,dt_valid_acc, label = 'Validation')
plt.legend()
plt.xlabel("Max Depth")
plt.ylabel("Accuracy")
plt.show


# ### Random Forest Models

# In[20]:


rf_train_acc = []
rf_valid_acc = []



for i in depth_range:
    np.random.seed(1)
    temp_forest = RandomForestClassifier(n_estimators = 100, max_depth = i)
    temp_forest.fit(X2_train,y2_train)
    temp_forest.score(X2_test, y2_test)
    rf_train_acc.append(temp_forest.score(X2_train, y2_train))
    rf_valid_acc.append(temp_forest.score(X2_test, y2_test)) 
    
    
    
    
rf_idx = np.argmax(rf_valid_acc)
rf_opt_depth = depth_range[rf_idx]

print('Optimal value for max_depth:          ',  "%.4f"% depth_range[rf_idx])
print('Training Accuracy for Optimal Model:  ',  "%.4f"% rf_train_acc[rf_idx])
print('Validation Accuracy for Optimal Model:',  "%.4f"% rf_valid_acc[rf_idx])


# In[21]:


plt.plot(depth_range,rf_train_acc, label = 'Training')
plt.plot(depth_range,rf_valid_acc, label = 'Validation')
plt.legend()
plt.xlabel("Max Depth")
plt.ylabel("Accuracy")
plt.show


# ### Evaluate Final Model
# I have decided to go with the Random Forest with the paramter max_depth = 24

# In[22]:


np.random.seed(1)

final_model = RandomForestClassifier(n_estimators = 100, max_depth = 24)
final_model.fit(X2_train,y2_train)



print('Training Accuracy for Final Model:  ', "%.4f"%  final_model.score(X2_train, y2_train))
print('Validation Accuracy for Final Model:', "%.4f"%  final_model.score(X2_test, y2_test))
print('Testing Accuracy for Final Model:   ',  "%.4f"% final_model.score(X2_hold, y2_hold))


# In[23]:


test_pred = final_model.predict(X2_test)
cm = confusion_matrix(test_pred,y2_test)


pd.DataFrame(cm, index=['<=50K', '>50K'], 
           columns=['Pred 1', 'Pred 2'] )


# In[24]:


print(classification_report(test_pred,y2_test))

