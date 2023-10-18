#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

data = pd.read_csv('WineQT.csv')
X = data[['fixed acidity', 'volatile acidity', 'citric acid','alcohol']]
y = data['quality'] 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Mean Squared Error:", mse)
print("R-squared:", r2)
new_data = np.array([[7.8,0.66, 0.22,9.1]])
quality = model.predict(new_data)
print("Quality is (0-10):", quality[0])


# In[3]:


print(data.head())


# In[4]:


print(data.tail())


# In[5]:


import matplotlib.pyplot as plt

# Assuming you have a list of qualities and their counts
qualities = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # Modify this as needed
quality_counts = [data['quality'].value_counts().get(quality, 0) for quality in qualities]

plt.bar(qualities, quality_counts)
plt.xlabel('Quality')
plt.ylabel('Count')
plt.title('Wine Quality Distribution')
plt.show()


# In[6]:


import matplotlib.pyplot as plt

# Example data
categories = ['Category A', 'Category B', 'Category C']
values = [10, 15, 7]

plt.bar(categories, values)
plt.xlabel('Categories')
plt.ylabel('Values')
plt.title('Bar Chart Example')
plt.show()


# In[7]:


import matplotlib.pyplot as plt

# Assuming you have a dataset you want to plot as a histogram
plt.hist(data['quality'], bins=11, edgecolor='k')  # 11 bins for quality values from 0 to 10
plt.xlabel('Quality')
plt.ylabel('Count')
plt.title('Quality Histogram')
plt.show()


# In[ ]:




