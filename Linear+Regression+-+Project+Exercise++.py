# $Id: Linear+Regression+-+Project+Exercise+.py 8163 2018-03-05 05:09:30Z milde $
# Author: Megan McClarty

"""
The following code comprises a brief project which analyzes a dataset called "Ecommerce Customers", runs a linear
regression model to determine the most impactful feature, and then makes a business recommendation based on the 
calculated coefficients between the Yearly Amount Spent by a customer and the customer's time spent on the website,
mobile app, and as a member in general.  This project is part of the Udemy Course "Python for Data Science and Machine
Learning". 

Linear Regression - Project Exercise

Congratulations! You just got some contract work with an Ecommerce company based in New York City that sells clothing 
online but they also have in-store style and clothing advice sessions. Customers come in to the store, have 
sessions/meetings with a personal stylist, then they can go home and order either on a mobile app or website 
for the clothes they want.

The company is trying to decide whether to focus their efforts on their mobile app experience or their website. 
They've hired you on contract to help them figure it out! Let's get started!

"""


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics


# ## Get the Data
# 
# We'll work with the Ecommerce Customers csv file from the company. It has Customer info, suchas Email, Address,
#  and their color Avatar. Then it also has numerical value columns:
# 
# * Avg. Session Length: Average session of in-store style advice sessions.
# * Time on App: Average time spent on App in minutes
# * Time on Website: Average time spent on Website in minutes
# * Length of Membership: How many years the customer has been a member.

# Bring in data csv as pandas dataframe
customers = pd.read_csv('Ecommerce Customers')
customers.head()
customers.describe()

# Using seaborn to run various visualizations of the data as jointplots and linear fit
joint = sns.jointplot(customers['Time on Website'],customers['Yearly Amount Spent'])

time_joint = sns.jointplot(customers['Time on App'],customers['Yearly Amount Spent'])

time_length = sns.jointplot(customers['Time on App'], customers['Length of Membership'], kind='hex')

# A pairplot shows the correlations between all features in the dataset
sns.pairplot(customers)
plt.show()

print('Based off the plotted correlations, the "Length of membership" looks to be the '
      'most correlated feature with Yearly Amount Spent.')

lin_cust = sns.lmplot('Length of Membership','Yearly Amount Spent',customers)


# ## Training and Testing Data
# 
# Split data into train and test sets
X = customers[['Avg. Session Length', 'Time on App',
       'Time on Website', 'Length of Membership']]
y = customers['Yearly Amount Spent']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)


# ## Training the Model

# Create a Linear Regression instance
lm = LinearRegression()

# Train and fit the Linear model on the training data
lm.fit(X_train,y_train)

# Determine coefficients of the model
lm.coef_
cdf = pd.DataFrame(lm.coef_,X.columns,columns=['Coeff'])
print(cdf)


# ## Predicting Test Data
predictions = lm.predict(X_test)

plt.scatter(y_test,predictions)
plt.show()


# ## Evaluating the Model
print('MAE: ' + str(metrics.mean_absolute_error(y_test,predictions)))
print('MSE: ' + str(metrics.mean_squared_error(y_test,predictions)))
print('RMSE: ' + str(np.sqrt(metrics.mean_squared_error(y_test,predictions))))


# ## Residuals
sns.distplot((y_test-predictions))
plt.show()


# ## Conclusion
# We still want to figure out the answer to the original question, do we focus our efforts
# on mobile app or website development?
print(cdf)


print('\n \n Coefficient Interpretation')

print('\n The attribute that has the strongest effect on Yearly Amount Spent is the '
      'Length of Membership.  Because this is not necessarily affected by a change to mobile or '
      'web service (insufficient data to claim there is a link), we should compare only the coefficients '
      'involving time on site/time on mobile app in making the decision.')

print('\n \n Do you think the company should focus more on their mobile app or on their website?')

print('\n In my opinion, the company should focus more on their mobile app, which has the second greatest coefficient after the '
      'Length of Membership.  Time on Website has an almost negligible effect, and it is possible efforts would be '
      'wasted to improve its impact.')

