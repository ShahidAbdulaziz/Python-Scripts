#!/usr/bin/env python
# coding: utf-8

# # DSCI 503 - Homework 07
# ### Shahid Abdulaziz

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot  as plt
from sklearn.linear_model import LinearRegression 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


# ## Problem 1: NYC Restaurants Dataset

# In[2]:


nyc = pd.read_csv('nyc.txt', sep = '\t')
nyc.head(10)


# In[3]:


X1 = nyc.iloc[:,1:].values
y1 = nyc.iloc[:,0].values


X_train_1, X_hold_1, y_train_1, y_hold_1 = train_test_split(X1, y1, test_size = 0.20, random_state=1)
X_valid_1, X_test_1, y_valid_1, y_test_1 = train_test_split(X_hold_1, y_hold_1, test_size = 0.50, random_state=1)


print('Training features:  ', X_train_1.shape)
print('Testing features:   ', X_test_1.shape)


# In[4]:


nyc_mod = LinearRegression()
nyc_mod.fit(X_train_1,y_train_1)

print('Intercept:       ', '%.2f'% nyc_mod.intercept_)
print('Coefficients:    ',  np.around(nyc_mod.coef_,2))


# In[5]:


print('Training r-Squared:', round(nyc_mod.score(X_train_1, y_train_1),4))
print('Training r-Squared:', round(nyc_mod.score(X_test_1, y_test_1),4))


# In[6]:



test_pred_1 = nyc_mod.predict(X_test_1)

print("Observed Prices:  ",y_test_1[:10])
print("Estimated Prices: ", np.around(test_pred_1[:10],2))


# In[7]:


nyc_new = pd.DataFrame({"Food":[22,18,25], "Decor":[12,19,22],"Service":[20,22,18],"Wait":[15,34,36],"East":[0,1,0]})
new_pred_1 = np.around(nyc_mod.predict(nyc_new ),2)

print("Estimated Prices: ", new_pred_1)


# ## Problem 2: Diamonds Dataset

# In[8]:


diamonds = pd.read_csv("diamonds.txt",sep = "\t")
diamonds.head(5)


# In[9]:


ln_carat = np.log(diamonds.carat)
ln_price = np.log(diamonds.price)

diamonds["ln_carat"] = ln_carat 
diamonds["ln_price"] = ln_price
diamonds.head(10)


# In[10]:


plt.figure(figsize = [10,4])

plt.subplot(1,2,1)
plt.scatter(diamonds.carat,diamonds.price, alpha = .2, s= 20)
plt.xlabel('Carat Size')
plt.ylabel('Price')
plt.title('Relationship between Price and Carat Size')


plt.subplot(1,2,2)
plt.scatter(diamonds.ln_carat,diamonds.ln_price, alpha = .2, s= 20)
plt.xlabel('"Natural Log of Carat Size')
plt.ylabel('Natural Log of Price')
plt.title('Relationship between Log-Price and Log-Carat Size')

plt.tight_layout()
plt.show


# In[11]:


X2 = diamonds.ln_carat.values.reshape(-1,1)
y2 = diamonds.ln_price.values




X_train_2 , X_test_2 , y_train_2 , y_test_2  = train_test_split(X2 , y2 , test_size = 0.10, random_state=1)
print('Training features Shape:  ', X_train_2 .shape)
print('Testing features Shape:   ', X_test_2 .shape)


# In[12]:


dia_mod = LinearRegression()
dia_mod.fit(X_train_2,y_train_2)

print('Intercept:    ', dia_mod.intercept_)
print('Coefficients: ', dia_mod.coef_)


# In[13]:


print('Training r-Squared:', round(dia_mod .score(X_train_2, y_train_2),4))
print('Training r-Squared:', round(dia_mod .score(X_test_2, y_test_2),4))


# In[14]:



test_pred_2 = dia_mod.predict(X_test_2)

print("Observed Prices:  ",np.around(np.exp(y_test_2[:10]),0))
print("Estimated Prices: ", np.around(np.exp(test_pred_2[:10]),0))


# In[15]:


diamonds_new = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0]).reshape(-1,1)
diamonds_new = np.log(diamonds_new)


# In[16]:


new_pred_2 = dia_mod.predict(diamonds_new)

print("Estimated Prices: ", np.around(np.exp(new_pred_1),0))


# ## Problem 3: Heart Disease Dataset

# In[17]:


hd = pd.read_csv('heart_disease.txt', sep = '\t')
hd.head(10)


# In[18]:


X3 = hd.iloc[:,:13].values
y3 = hd.iloc[:,13].values


X_train_3 , X_test_3 , y_train_3 , y_test_3  = train_test_split(X3 , y3 , test_size = 0.20, random_state=1, stratify=y3)



print('Training features Shape:  ', X_train_3 .shape)
print('Testing features Shape:   ', X_test_3 .shape)


# In[19]:



hd_mod = LogisticRegression(solver='lbfgs', penalty='none', multi_class='multinomial', max_iter =10000)
hd_mod.fit(X_train_3,y_train_3)

print('Intercept:    ', hd_mod.intercept_)
print('Coefficients: ','\n'
      ,'             ', np.around(hd_mod.coef_,4))


# In[20]:


print('Training Accuracy:  ', "%.4f"% hd_mod.score(X_train_3, y_train_3))
print('Validation Accuracy:', "%.4f"% hd_mod.score(X_test_3, y_test_3))


# In[21]:


test_pred_3 = hd_mod.predict(X_test_3)


print("Observed Labels:",y_test_3[:20])
print("Predicted Labels:",test_pred_3[:20])


# In[22]:


pd.DataFrame(hd_mod.predict_proba(X_test_3), columns = hd_mod.classes_).head(10)


# ## Problem 4: Gapminder Dataset

# In[23]:


gm = pd.read_csv('gapminder_data.txt', sep = '\t')
gm = gm[gm.year==2018]
gm.head(10)


# In[24]:


X4 = gm.iloc[:,4:].values
y4 = gm.iloc[:,2].values


X_train_4 , X_test_4 , y_train_4 , y_test_4  = train_test_split(X4 , y4 , test_size = 0.20, random_state=1,stratify=y4)



print('Training features Shape:  ', X_train_4 .shape)
print('Testing features Shape:   ', X_test_4 .shape)


# In[25]:



gm_mod = LogisticRegression(solver='lbfgs', penalty='none', multi_class='multinomial', max_iter =1000)
gm_mod.fit(X_train_4,y_train_4)

print('Intercept:    ', gm_mod.intercept_)
print('Coefficients: ','\n', 
      '              ', gm_mod.coef_)


# In[26]:


print('Training Accuracy:  ', "%.4f"% gm_mod.score(X_train_4, y_train_4))
print('Validation Accuracy:', "%.4f"% gm_mod.score(X_test_4, y_test_4))


# In[27]:


test_pred_4 = gm_mod.predict(X_test_4)


print("Observed Labels:",y_test_4[:8])
print("Predicted Labels:",test_pred_4[:8])


# In[28]:


pd.DataFrame(gm_mod.predict_proba(X_test_4), columns = gm_mod.classes_).head(10)


# In[29]:


gm_new = pd.DataFrame({'life_exp':[75,75,75,75,75,75], 'gdp_per_cap':[5000,5000,5000,20000,20000,20000],'gini':[30,40,50,30,40,50]})

pd.DataFrame(np.around(gm_mod.predict_proba(gm_new),3), columns = gm_mod.classes_)


# According to our model:
#    * Country 0 is most likely in Europe.
#    * Country 1 is most likely in Africa.
#    * Country 2 is most likely in Africa.
#    * Country 3 is most likely in Europe.
#    * Country 4 is most likely in Asia.
#    * Country 5 is most likely in Americas.
