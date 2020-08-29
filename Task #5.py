#!/usr/bin/env python
# coding: utf-8

# # Task 5: Exploratory Data Analysis

# ### Problem Statement:
# You are the business owner of the retail firm and want to see
# how your company is performing. You are interested in finding
# out the weak areas where you can work to make more profit.
# What all business problems you can derive by looking into the
# data?
# 

# ### Solution:

# In[136]:


# Let us first import the modules

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
from matplotlib import pyplot as plt


# In[137]:


# loading our dataset
df = pd.read_csv('C:\\Users\\Tejasvi Jain\\Downloads\\SampleSuperstore.csv')
df.head()


# In[138]:


# Dropping the postal code
df=df.drop(columns='Postal Code',axis=1)


# In[139]:


# Checking for any missing values
df.isnull().sum()


# In[140]:


# Checking the datatype
df.info()


# In[141]:


# To get the summary of all the cols
# Here we will get the NaN for the summary field which are not appropriate for that datatype
# of the column
df.describe(include='all')


# In[142]:


# Let us get the correlation in the data

# Compute the correlation matrix
corr = df.corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})


# ##### From the above correlation matrix we can tell there is a very strong direct correlation between the sales and the profit i.e., more sales, more profit and also a strong indirect correlation between profit and the discount provided i.e., more discount less profit

# ### Data Visualisation

# Let's find out some more about the relationships observed through the correlation matrix. 

# In[143]:


# Range of profit and loss
Profit_column=df['Profit']
plt.plot(Profit_column)
plt.title('Profit Loss Spectrum')


# In[144]:


print('Maximum Profit:',Profit_column.max())


# In[145]:


print('Maximum Loss:',Profit_column.min())


# From Here we notice that there is a wide range of profits and losses can amount upto 6599 dollars.

# In[146]:


# Checking for losses
df['Profit']<=0


# In[147]:


# Loss wise data distribution
Loss=df[df['Profit']<=0]
Loss


# ##### Let's check any relation between categorial variables:

# In[148]:


# Segment Distribution
sns.set_style('darkgrid')
sns.countplot(x='Segment',data=df)


# In[149]:


# Profits/Losses of each segment
Segment_sum=Loss.loc[:,["Segment","Profit"]]
Segment_sum=Segment_sum.groupby(['Segment'], as_index=False).sum()
sns.barplot(x=Segment_sum['Segment'],y=Segment_sum['Profit'])


# In[150]:


Segment_sum.head()


# ##### Hence we can tell the maximum demand is from consumer segment and the maximum profit and loss generated is also from the comsumer segment.

# In a similar way let's also check loss making shipping mode:

# In[151]:


# SHipping Mode 
sns.set_style('darkgrid')
sns.countplot(x='Ship Mode',data=df)


# In[152]:


Segment_sum=Loss.loc[:,["Ship Mode","Profit"]]
Segment_sum=Segment_sum.groupby(['Ship Mode'], as_index=False).sum()
sns.barplot(x=Segment_sum['Ship Mode'],y=Segment_sum['Profit'])


# In[153]:


Segment_sum


# ##### Hence we can see that the standard shipping mode proves to be inefficient means.

# Now let's look at the profit and sales data of categories and sub-categories

# In[154]:


# Checking for types of categories and sub-categories
df['Category'].unique()


# In[155]:


df['Sub-Category'].unique()


# In[156]:


categories_all=df.loc[:,['Category','Sub-Category','Sales','Quantity','Profit','Discount']]
categories_all


# In[157]:


category_diff=categories_all.groupby(['Category'],as_index=False).sum()
category_diff


# #### From all the above information we can conclude that furniture despite having good number of sales generates very little profit as compared to technology that generates very high profits with almost similar number of sales

# Since Furniture is the loss making category let's explore more of it's sub-categories:
# 

# In[158]:


# Exploring Sub-categories of furniture
furniture_all=categories_all[categories_all['Category']=='Furniture']
furniture_all.groupby(['Sub-Category'],as_index=False).sum()


# #### Hence we can clearly observe that Bookcases and Tables generates losses though the sales number is high.

# In[164]:


# Exploring Sub-categories of Technology
furniture_all=categories_all[categories_all['Category']=='Technology']
furniture_all.groupby(['Sub-Category'],as_index=False).sum()


# #### It can be clearly seen that technology generates highest profit with highest number of sales and lower discount compared to furniture and office supplies

# Now let's also analyse the impact of discount provided on profit by various products:

# In[159]:


# Bivariate Analysis
sns.pairplot(df[0:10000], x_vars=['Category', 'Profit', 'Quantity'], y_vars='Discount',
             size=7, aspect=0.7, kind='scatter')


# ##### Therefore Bivariate analysis of discount with categories, profit and quantity tells the same story. Most of the discount is provided on furniture and losses are more when higher discount is provided even though quantities sold are high.

# Let's analyse the sales and profit data for the different States and Cities:

# In[160]:


States_cities=df.loc[:,['State','City','Sales','Quantity','Profit']]
States_cities


# In[161]:


# Visualising profit data of States
plt.figure(figsize=(10,10),dpi=110)
States_sales_profit=States_cities.groupby(['State'],as_index=False).sum()
sns.barplot(y=States_sales_profit['State'],x=States_sales_profit['Profit'])


# Let's futher analyse the loss making states:

# In[162]:


# Aggregating Loss making states
States_loss=States_sales_profit[States_sales_profit['Profit']<=0]
States_loss


# #### From this we can observe Texas is the most loss making State with the highest quantity sold followed by Ohio and Pennsylvania

# ## CONCLUSION:

# **1. Standard shipping mode is the most inefficient means which can be looked after to eliminate the factors causing losses.**

# **2. Bookcases and tables in the Sub-categories can be removed which can help cut losses. Since the sales of these products are high, we can work with the discount provided on such products to make use of high sales figure.**

# **3.We can also Remove the entire loss making sub categories and divert the resources to the highest profit making categories and subcategories which are Technology.**

# **3. We also conclude that Texas is the largest loss making state followed by Ohio and Pennsylvania which can also be looked into and cut ineffieciencies causing losses.**

# In[ ]:




