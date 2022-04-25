#!/usr/bin/env python
# coding: utf-8

# # DSCI 503 - Homework 04
# ### Shahid Abdulaziz

# In[1]:


import numpy as np
import math as mt
import matplotlib.pyplot as plt


# ## Problem 1: Sample Mean and Variance

# In[2]:


x = [10, 16, 26, 12, 17, 22, 14, 12, 21, 16]
n = len(x)

mean = np.sum(x)/n
diff = [x-mean]
var = np.sum(np.power(diff,2))/n


print("Sample Mean:     "+str("%.2f"% mean))
print("Sample Variance: "+ str((var)))


# In[3]:


mean = np.mean(x)
var = np.var(diff, ddof = 1)

print("Sample Mean:     "+str("%.2f"% mean))
print("Sample Variance: "+ str(("%.2f"%var)))


# ## Problem 2: Scoring a Regression Model

# In[4]:


def find_sse(true_y, pred_y):
    SSE = np.sum(np.power(np.subtract(true_y, pred_y),2))
    
    return SSE
    
    


# In[5]:


true_y = [22.1, 17.9, 16.5, 14.3, 19.8, 23.7, 22.0, 18.4, 25.7, 19.2]
pred_1 = [21.4, 16.7, 17.9, 12.1, 22.1, 25.1, 21.7, 19.3, 23.4, 19.9]
pred_2 = [20.7, 18.1, 16.9, 13.6, 21.9, 24.8, 20.3, 21.1, 24.8, 18.4]

sse_1 = find_sse(true_y, pred_1)
sse_2 = find_sse(true_y,pred_2)

print("Model 1 SSE: "+ str("%.2f"%  sse_1))
print("Model 2 SSE: "+ str("%.2f"%  sse_2))


# ## Problem 3: Classification Model

# In[6]:


def find_accuracy(true_y, pred_y):
    accurate = np.sum(np.array(true_y) == np.array(pred_y))
    length = len(true_y)
    score = (accurate/length)*100
    
 
    
    return score


# In[7]:


true_diag =  ['P', 'P', 'N', 'N', 'P', 'N', 'N', 'N', 'P', 'N', 'N', 'N', 'N', 'P', 'P', 'N', 'N',
             'N', 'N', 'N']

pred_diag =  ['N', 'P', 'N', 'P', 'P', 'N', 'P', 'N', 'P', 'N', 'N', 'N', 'P', 'P', 'P', 'N', 'N',
             'N', 'P', 'N']

print("Model Accuracy:", "%.2f" % find_accuracy(true_diag, pred_diag))


# In[8]:


true_labels = ['dog', 'dog', 'cat', 'dog', 'cat', 'cat', 'cat', 'dog', 'cat', 'cat', 'dog', 'cat',
 'cat', 'dog', 'dog', 'dog', 'dog', 'cat', 'cat', 'cat', 'dog', 'dog', 'cat', 'cat']

pred_labels =  ['dog', 'dog', 'cat', 'dog', 'cat', 'dog', 'cat', 'dog', 'cat', 'cat', 'dog', 'cat',
 'cat', 'dog', 'cat', 'dog', 'dog', 'cat', 'dog', 'cat', 'dog', 'dog', 'cat', 'cat']

print("Model Accuracy:",  "%.2f" % find_accuracy(true_labels , pred_labels))


# ## Problem 4: Classification Report

# In[9]:


def classification_report(true_y, pred_y):
    classes = []
    classes = np.unique(true_y)
    
    Accuracy = find_accuracy(true_y, pred_y)
    TP =  (np.sum( ( np.array(true_y) == np.array(pred_y)) & (np.array(true_y) == classes[1]))  / np.sum(np.array(true_y) == classes[1]))*100
    FP =  (np.sum( ( np.array(true_y) == np.array(pred_y)) & (np.array(true_y) == classes[1]))  / np.sum(np.array(pred_y) == classes[1]))*100
    TN =  (np.sum( ( np.array(true_y) == np.array(pred_y)) & (np.array(true_y) == classes[0]))  / np.sum(np.array(true_y) == classes[0]))*100
    FN =  (np.sum( ( np.array(true_y) == np.array(pred_y)) & (np.array(true_y) == classes[0]))  / np.sum(np.array(pred_y) == classes[0]))*100
    print("Positive Class:     "+str(classes[1])     )
    print("Negative Class:     "+str(classes[0])+"\n")
    
    print("Accuracy:           "+str("%.2f"% Accuracy))
    print("Positive Precision: "+str("%.2f"% TP))
    print("Positive Recall:    "+str("%.2f"% FP))
    print("Negative Precision: "+str("%.2f"% TN))
    print("Negative Precision: "+str("%.2f"% FN))
    


# In[10]:


classification_report(true_diag,pred_diag  )


# In[11]:


classification_report(true_labels,pred_labels )


# ## Problem 5: Transformation of Random Variables

# In[25]:


np.random.seed(1)


X = np.random.normal(0,.4,25000)
Y = np.exp(x)

print("Sample Mean of X: "+str("%.4f"% np.mean(x)))
print("Sample Mean of X:  "+str("%.4f"% np.std(x, ddof =1)))
print("Sample Mean of Y:  "+str("%.4f"% np.mean(Y)))
print("Sample Mean of X:  "+str("%.4f"% np.std(x, ddof =1)))


# In[13]:



plt.figure(figsize=[12,4])
plt.subplot(1, 2, 1)
plt.hist(x, edgecolor = 'black', bins = 30, color = 'red' )
plt.title("Histogram of X Values")


plt.subplot(1, 2, 2)
plt.hist(Y, edgecolor = 'black', bins = 30, color = 'blue' )
plt.title("Histogram of Y Values")



plt.show


# In[29]:


figure, axes = plt.subplots(nrows=1, ncols=2, figsize=[12, 4])

axes[0].hist(X, bins=30, edgecolor='black', color='pink',)
axes[0].set_title('Histrogram of X Values')
axes[1].hist(Y, bins=30, edgecolor='black', color='yellow',)
axes[1].set_title('Histrogram of Y Values')
plt.show()


# In[14]:


print("Probability that Y is less than 0.5: "+ str("%.4f"% np.mean(Y < 0.5)))
print("Probability that Y is less than 1.0: "+ str("%.4f"% np.mean(Y < 1.0)))
print("Probability that Y is less than 2.0: "+ str("%.4f"% np.mean(Y < 2.0)))



# ## Problem 6: Stochastic Linear Relationships

# In[15]:


np.random.seed(1) 

x_vals = np.random.normal(10,2,200)
errors = np.random.normal(0,1.2,200)
y_vals = 5.1 +.9 * x_vals +errors


plt.figure(figsize=[8,6])
plt.scatter(x=x_vals , y=y_vals , s=60, alpha=0.8, 
            color='red', edgecolor='black')
plt.xlabel('X Values')
plt.ylabel('Y Values')
plt.show()


# In[16]:


diff_x = np.subtract(np.mean(x_vals),x_vals)
diff_y = np.subtract(np.mean(y_vals),y_vals)



print("Correlation between X and Y:", str("%.4f"% ((np.sum(np.multiply(diff_x, diff_y)))/(np.sqrt(np.sum(np.power(diff_x,2))*np.sum(np.power(diff_y,2)))))))


# ## Problem 7:  Relationship between Life Expectancy and Per Capita GDP

# In[17]:


import pandas as pd
df = pd.read_csv('gapminder_data.txt', sep='\t')
country = df.country.values
year = df.year.values
continent = df.continent.values
population = df.population.values
life_exp = df.life_exp.values
pcgdp = df.gdp_per_cap.values
gini = df.gini.values


# In[18]:


continent_list = ['africa', 'americas', 'asia', 'europe' ]
color_list = ['red','blue','green','orange']


# In[19]:




for i in range(0,len(continent_list)):

    sel = np.array((year == 2018) ) & np.array(continent == continent_list[i])
    
    
    current_continent = continent_list[i]
    

    
    plt.scatter(x=np.log(pcgdp[sel]),y=life_exp[sel] , s=100, alpha=0.7, 
    color=color_list[i], edgecolor='black', label = current_continent.title())
    plt.title('Life Expectency vs Per Capita GDP (2018)')
    plt.xlabel('Natural Log of Per Capita GDP')
    plt.ylabel('Life Expectancy')
    plt.legend()
plt.show  
      
      
      
        

  


# In[20]:


for i in range(0,len(continent_list)):

    sel = np.array((year == 2018) ) & np.array(continent == continent_list[i])
    
    
    current_continent = continent_list[i]
    
    plt.figure(figsize=[10,8])
    
    for i in range(1,5):
        plt.subplot()
        plt.scatter( x=np.log(pcgdp[sel]),y=life_exp[sel] , s=100, alpha=0.7, edgecolor='black')
        plt.title('Life Expectancy vs Per Capita GDP ' + current_continent.capitalize())
        plt.xlabel('Natural Log of Per Capita GDP')
        plt.ylabel('Life Expectancy')
        plt.xlim([6, 12])
        plt.ylim([45, 90])
        plt.tight_layout()
        plt.show  
      


# ## Problem 8: Trends by Country

# In[21]:


year_range = list(range(1799, 2018))

USData = df[(df.country == "United States") & (df.year >= 1800) & (df.year <= 2018)]
VenezuelaData = df[(df.country == "Venezuela")& (df.year >= 1800) & (df.year <= 2018)]
VietnamData = df[(df.country == "Vietnam")& (df.year >= 1800) & (df.year <= 2018)]
ZambiaData = df[(df.country == "Zambia")& (df.year >= 1800) & (df.year <= 2018)]
ZimbabweData = df[(df.country == "Zimbabwe")& (df.year >= 1800) & (df.year <= 2018)]



plt.figure(figsize=[8,4])
plt.plot(year_range, USData.population, lw=2, label='United States')
plt.plot(year_range, VenezuelaData.population, lw=2, label='Venezuela')
plt.plot(year_range, VietnamData.population, lw=2, label='Vietnam')
plt.plot(year_range, ZambiaData.population, lw=2, label='Zambia')
plt.plot(year_range, ZimbabweData.population, lw=2, label='Zimbabwe')
plt.plot([1800, 2019], [0,0], ls='--', color='black')
plt.legend()
plt.xlabel('Years')
plt.ylabel('Population')
plt.title('Population by Year')
plt.show()


# In[22]:



plt.figure(figsize=[8,4])
plt.plot(year_range, USData.life_exp, lw=2, label='United States')
plt.plot(year_range, VenezuelaData.life_exp, lw=2, label='Venezuela')
plt.plot(year_range, VietnamData.life_exp, lw=2, label='Vietnam')
plt.plot(year_range, ZambiaData.life_exp, lw=2, label='Zambia')
plt.plot(year_range, ZimbabweData.life_exp, lw=2, label='Zimbabwe')
plt.plot([1800, 2019], [0,0], ls='--', color='black')
plt.legend()
plt.xlabel('Years')
plt.ylabel('Life Expectancy')
plt.title('Life Expectancy by Year')
plt.show()

