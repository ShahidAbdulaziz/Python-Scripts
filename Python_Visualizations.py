#!/usr/bin/env python
# coding: utf-8

# # DSCI 503 - Homework 05
# ### Shahid Abdulaziz

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ## Problem 1: Working with 2D Arrays

# In[2]:


np.random.seed(1)
z = np.random.uniform(0,10,[5,8])
z = np.around(z,2)

print(z)


# In[3]:


print("Row 3:    "+ str(z[2]))
print("Column 6: "+ str(z[:,5]))


# In[4]:


print("Row Sums:    "+ str(np.sum(z,axis =1)))
print("Column Sums: "+ str(np.sum(z,axis =0)))
print("Array Sum:   "+ str(np.sum(z)))


# # Problem 2: Reshaping and Stacking Arrays

# In[5]:


np.random.seed(167)

x1 = np.random.normal(50,10,1000)
x2 = np.random.normal(20,5,1000)
x3 = np.random.normal(100,30,1000)
x4 = np.random.normal(10,2,1000)


x1 = x1.reshape(x1.shape[0],-1)
x2 = x2.reshape(x2.shape[0],-1)
x3 = x3.reshape(x3.shape[0],-1)
x4 = x4.reshape(x4.shape[0],-1)

X = np.hstack((x1,x2,x3,x4))

X = np.around(X, 2) 

print(X.shape)


# In[6]:


print(X[0:6])


# # Problem 3: Standardization

# In[7]:


Xmean = np.mean(X, 0)
Xstd  = np.std(X, 0)

print("Column means:               "+ str(np.around(Xmean,2)))
print("Column standard deviations: "+ str(np.around(Xstd,2 )))


# In[8]:


W = np.divide(np.subtract(X, Xmean),Xstd)
Wmean = np.mean(W,axis = 0)
Wstd  = np.std(W,axis = 0)

print("Column means:               "+ str(np.around(Wmean,2)))
print("Column standard deviations: "+ str(np.around(Wstd,2)))


# # Problem 4: Load Auto MPG Dataset

# In[9]:


auto = pd.read_csv('auto_mpg.txt', sep='\t')
auto.head(10)


# In[10]:


print(auto.shape)


# In[11]:


print(np.mean(auto))


# # Problem 5: Regional Counts and Means

# In[12]:


regions = sorted(pd.unique(auto.region))
print(regions)


# In[13]:


eur_auto  = auto[(auto.region=='eur')]
asia_auto = auto[(auto.region=='asia')]
usa_auto  = auto[(auto.region=='usa')]

print("Number of cars manufactured in Asia:  ", len(asia_auto))
print("Number of cars manufactured in Europe:", len(eur_auto))
print("Number of cars manufactured in USA:   ", len(usa_auto))


# In[14]:


eur_means = np.mean(eur_auto)
asia_means = np.mean(asia_auto)
usa_means = np.mean(usa_auto)


mean_df=pd.DataFrame(
    {'region':['asia','eur','usa'],
     'mpg':[asia_means['mpg'],
            eur_means['mpg'],
            usa_means['mpg']],
      'cyl':[asia_means['cyl'],
             eur_means['cyl'],
             usa_means['cyl']],
       'wt':[asia_means['wt'],
             eur_means['wt'],
             usa_means['wt']]})

mean_df.set_index(['region'],inplace=True)

mean_df


# # Problem 6: Average Weight and MPG by Region

# In[15]:


colors1 = ['black','red','yellow']

plt.figure(figsize=[8,4])
plt.subplot(1, 2, 1)

plt.bar(x=regions, height=mean_df.mpg, color=colors1, edgecolor='black')
plt.xlabel('Region')
plt.ylabel('Average MPG')
plt.title('Average MPG by Region')

plt.subplot(1, 2, 2)
plt.bar(x=regions, height=mean_df.wt, color=colors1, edgecolor='black')
plt.xlabel('Region')
plt.ylabel('Average Weight in Pounds')
plt.title('Average Weight by Region')

plt.tight_layout() 
plt.show()


# # Problem 7: Relationship between Weight and Miles Per Gallon

# In[16]:



plt.figure(figsize = [12,4])

for i in range(0,len(regions)):
    plt.subplot(1, 3, i+1) 
    BMask = auto[auto.region==regions[i]]
    plt.scatter(BMask['wt'],BMask['mpg'],edgecolor='black',alpha = .8, color = colors1[i])
    plt.xlabel("Weight in Pounds")
    plt.ylabel("Miles Per Gallon")
    plt.title("Weight vs MPG "+ regions[i].capitalize())  
    plt.xlim(1200,5000)
    plt.ylim(0,50)
    
plt.tight_layout()
plt.show()

 


# # Problem 8: Cylinder Distribution by Region

# In[17]:


cyl_values = np.unique(auto.cyl)


# In[18]:


cyl_counts_by_region = pd.crosstab(auto.cyl,auto.region)
   
cyl_counts_by_region


# In[19]:


cyl_props_by_region = np.divide(cyl_counts_by_region, np.sum(cyl_counts_by_region, axis = 0))

bar_bottoms = np.cumsum(cyl_props_by_region) - cyl_props_by_region

colors2 = ['black','red','yellow','orange','purple']


cyl_props_by_region


# In[20]:


bar_bottoms = [0,0,0]
idx = 0
plt.figure(figsize = [6,4])

for index, row in cyl_props_by_region.iterrows() :
    plt.bar(regions, row, color=colors2[idx], bottom = bar_bottoms, label = row.name)
    bar_bottoms += row
    idx += 1
plt.legend(bbox_to_anchor=(1.15, 1.0))
plt.title("Distribution of Cylinder Numbers by Region")
plt.xlabel("Region")
plt.ylabel("Proportion")
plt.show()


# In[26]:


cyl_props_by_region.index

