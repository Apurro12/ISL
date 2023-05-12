#!/usr/bin/env python
# coding: utf-8

# In[70]:


from sklearn.linear_model import LinearRegression
from IPython.display import Image
import numpy as np
from matplotlib import pyplot as plt


# In[2]:


num_data = 50


# In[3]:


#Initial points
x = np.linspace(0,100, num_data)


# # From the next model we can see that is perfect, and there is nothing to interpret

# In[4]:


y_0 = np.linspace(0,100, num_data)

model = LinearRegression()
model.fit(x.reshape(-1,1),y_0)

beta_0, beta_1 = model.intercept_, model.coef_[0]


# In[5]:


plt.scatter(x,y_0)
plt.title(f"The perfect LR: beta_0 = {np.round(beta_0,5)} , beta_1 = {np.round(beta_1,5)}")
plt.show()


# # But... what happens when we fit a model for data that is not correlated

# In[6]:


y_20 = np.linspace(0,10, num_data) + np.random.normal(size = num_data, scale = 20)

model = LinearRegression()
model.fit(x.reshape(-1,1),y_20)

beta_0, beta_1 = model.intercept_, model.coef_[0]


# In[7]:


plt.title(f"A Descorellated plot: beta_0 = {np.round(beta_0,5)} , beta_1 = {np.round(beta_1,5)}")
plt.scatter(x,y_20)
plt.show()


# ### The question arises, this beta_1 and beta_0, how much confidence they deserve?

# In[8]:


# If I continue gathering data from this, I could create an histogram from that


# In[92]:


#One way could be calculate the STD, but after using the formula lets do it as always
Image(filename = "img/img_0.png", width=1000, height=1000)


# In[93]:


list_beta_1 = []

num_experiments = 5000
for current_experiment in range(num_experiments):
    x = np.random.rand(num_data)
    y = x + np.random.normal(size = num_data, scale = 20)
    
    model = LinearRegression()
    model.fit(x.reshape(-1,1),y_0)
    
    list_beta_1.append(model.coef_[0])


# In[94]:


list_beta_1_mean = np.mean(list_beta_1)
list_beta_1_std = np.std(list_beta_1)


# In[95]:


plt.title(f"Distribution of b_1: mean = {np.round(list_beta_1_mean,2)}, std = {np.round(list_beta_1_std,3)}")
plt.hist(list_beta_1, bins = 50, density = True)
plt.show()

