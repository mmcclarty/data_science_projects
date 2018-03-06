# $Id: Ecommerce_Linear_Regression_Analysis.py 8163 2018-03-05 05:09:30Z milde $
# Author: Megan McClarty

"""
The following code comprises a brief project which analyzes a dataset called "Ecommerce Customers", runs a linear
regression model to determine the most impactful feature, and then makes a business recommendation based on the 
calculated coefficients between the Yearly Amount Spent by a customer and the customer's time spent on the website,
mobile app, and as a member in general.  This project is part of the Udemy Course "Python for Data Science and Machine
Learning". 

### This version has been generalized for different data sets ###

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


def get_df(data_file):
      '''
      Read csv file of data and return a pandas dataframe.
      
      :param data_file: Local path to csv data file
      :return: 
      '''

      customers = pd.read_csv(data_file)

      return customers


def visualize_data(customers, target, features):
      '''
      Produce some Seaborn plots for data visualization of key features
      
      :param customers: dataframe 
      :param target: Identified target column (string)
      :param features: Identified feature columns (list)
      :return: 
      '''

      # Using seaborn to run various visualizations of the data as jointplots and linear fit
      joint = sns.jointplot(customers[features[0]], customers[target])

      time_joint = sns.jointplot(customers[features[1]], customers[target])

      time_length = sns.jointplot(customers[features[1]], customers[features[2]], kind='hex')

      # A pairplot shows the correlations between all features in the dataset
      sns.pairplot(customers)
      plt.show()

      lin_cust = sns.lmplot(features[2], target, customers)


def linear_reg_train(customers, target, features):
      '''
      Split data into train and test sets, fit the data to a linear regression model.
      
      :param customers: dataframe
      :param target: Identified target column (string)
      :param features: Identified features columns (list)
      :return: 
      '''

      # Split data into train and test sets
      X = customers[[features[0], features[1], features[2], features[3]]]
      y = customers[target]

      # Split into train and test with test size of 40%
      X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)

      # ## Training the Model
      # Create a Linear Regression instance
      lm = LinearRegression()

      # Train and fit the Linear model on the training data
      lm.fit(X_train, y_train)

      # Determine coefficients of the model
      lm.coef_
      cdf = pd.DataFrame(lm.coef_,X.columns,columns=['Coeff'])
      print(cdf)

      return lm, X, y, X_test, y_test


def predictions(lm, X_test, y_test):
      '''
      Make predictions of test set using the linear model developed.
      
      :param lm: Linear regression instance
      :param X_test: Test set of features
      :param y_test: Test set of target
      :return: 
      '''

      # ## Predicting Test Data
      predictions = lm.predict(X_test)

      plt.scatter(y_test, predictions)
      plt.show()

      return predictions

def model_evaluation(y_test, predictions):
      '''
      Evaluate the accuracy of the model employed.
      
      :param y_test: Test set of target data
      :param predictions: Predicted targets using model
      :return: 
      '''

      # Evaluate the model
      print('MAE: ' + str(metrics.mean_absolute_error(y_test, predictions)))
      print('MSE: ' + str(metrics.mean_squared_error(y_test, predictions)))
      print('RMSE: ' + str(np.sqrt(metrics.mean_squared_error(y_test, predictions))))

      # Residuals
      sns.distplot((y_test-predictions))
      plt.show()

      # We still want to figure out the answer to the original question, do we focus our efforts
      # on mobile app or website development?

      print('\n \n Coefficient Interpretation')

      print('\n The attribute that has the strongest effect on Yearly Amount Spent is the '
            'Length of Membership.  Because this is not necessarily affected by a change to mobile or '
            'web service (insufficient data to claim there is a link), we should compare only the coefficients '
            'involving time on site/time on mobile app in making the decision.')

      print('\n Do you think the company should focus more on their mobile app or on their website?')

      print('\n In my opinion, the company should focus more on their mobile app, which has the second greatest coefficient after the '
            'Length of Membership.  Time on Website has an almost negligible effect, and it is possible efforts would be '
            'wasted to improve its impact.')


def main():
      '''
      
      :return: 
      '''

      # We'll work with the Ecommerce Customers csv file from the company. It has Customer info, suchas Email, Address,
      #  and their color Avatar. Then it also has numerical value columns:
      #
      # * Avg. Session Length: Average session of in-store style advice sessions.
      # * Time on App: Average time spent on App in minutes
      # * Time on Website: Average time spent on Website in minutes
      # * Length of Membership: How many years the customer has been a member.

      # Bring in data csv as pandas dataframe
      chosen_file = 'Ecommerce Customers'
      customers = get_df(chosen_file)

      # Look at snippet of dataframe
      print(customers.head())

      # Choose target and features from columns
      target = 'Yearly Amount Spent'
      features = ['Time on Website', 'Time on App', 'Length of Membership', 'Avg. Session Length']

      # Visualize data and correlations between features
      visualize_data(customers, target, features)
      print('Based off the plotted correlations, the "Length of membership" looks to be the '
            'most correlated feature with Yearly Amount Spent.')

      # Split into train and test data and fit linear regression
      lm, X, y, X_test, y_test = linear_reg_train(customers, target, features)

      # Predict results of test data
      prediction = predictions(lm, X_test, y_test)

      # Evaluate model
      model_evaluation(y_test, prediction)


print('\n This application takes a dataset and performs a linear regression analysis on it. \n')
main()








