'''
This is an export of a Jupyter notebook file.

It performs a Linear Regression analysis on two housing price datasets.

The 'USA_Housing.csv' dataset is bundled with this file.  The Boston-Housing dataset is an inbuilt dataset bundled
with scikit-learn.

'''

# In[3]:


import pandas as pd
import numpy as np 


# In[4]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[5]:


get_ipython().magic(u'matplotlib inline')


# In[6]:


df = pd.read_csv('USA_Housing.csv')


# In[7]:


df.head()


# In[8]:


df.describe()


# In[15]:


df.columns


# In[10]:


sns.pairplot(df)


# In[11]:


sns.distplot(df['Price'])


# In[13]:


sns.heatmap(df.corr(),annot=True)


# In[16]:


# Start Linear Regression model
X = df[['Avg. Area Income', 'Avg. Area House Age',
       'Avg. Area Number of Rooms', 'Avg. Area Number of Bedrooms',
       'Area Population']]


# In[17]:


# Identify the target
y = df['Price']


# In[19]:


from sklearn.cross_validation import train_test_split


# In[20]:


# Split the dataset into training and test sets with 40% test size
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.40, random_state=101)


# In[21]:


# Now set up the linear regression model itself
from sklearn.linear_model import LinearRegression 


# In[22]:


lm = LinearRegression()


# In[23]:


# Fit the model to your training data
lm.fit(X_train, y_train)


# In[24]:


print(lm.intercept_)


# In[25]:


lm.coef_


# In[26]:


X.columns


# In[27]:


cdf = pd.DataFrame(lm.coef_,X.columns,columns=['Coeff'])


# In[29]:


# An average one-unit increase in the value of the columns is associated with a corresponding increase in the Price (in $)
cdf


# In[30]:


# Repeat this analysis with a more accurate (real) dataset
from sklearn.datasets import load_boston


# In[31]:


boston = load_boston()


# In[34]:


boston.keys()


# In[39]:


# Turn the in-built dataset into a pandas dataframe
bostondf = pd.DataFrame(boston.data,columns=boston.feature_names)
bostondf['target'] = pd.Series(boston.target)


# In[42]:


bostondf.head()


# In[44]:


# Start Linear Regression model
X = bostondf[['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS',
       'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']]
y = bostondf['target']


# In[45]:


# Make Boston training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.40, random_state=101)


# In[46]:


# Make linear regression fit
lm.fit(X_train,y_train)


# In[47]:


cdf = pd.DataFrame(lm.coef_,X.columns,columns=['Coeff'])


# In[48]:


# For a one unit increase in the left-hand column values, the price will change by the coeff indicated (in thousands $)
cdf

