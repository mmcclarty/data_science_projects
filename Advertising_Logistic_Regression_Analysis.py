# $Id: Advertising_Logistic_Regression.py 8163 2018-03-08 05:09:30Z milde $
# Author: Megan McClarty

"""
Logistic Regression Project

The following code comprises a brief project which analyzes a dataset called "Advertising", runs a logistic
regression model to categorize users into whether they Clicked on an ad or not.  
This project is part of the Udemy Course "Python for Data Science and Machine
Learning". 

### This version has been generalized for different data sets ###

In this project we will be working with a fake advertising data set, indicating whether or not a particular internet 
user clicked on an Advertisement. We will try to create a model that will predict whether or not they will click on 
an ad based off the features of that user.

This data set contains the following features:

* 'Daily Time Spent on Site': consumer time on site in minutes
* 'Age': cutomer age in years
* 'Area Income': Avg. Income of geographical area of consumer
* 'Daily Internet Usage': Avg. minutes a day consumer is on the internet
* 'Ad Topic Line': Headline of the advertisement
* 'City': City of consumer
* 'Male': Whether or not consumer was male
* 'Country': Country of consumer
* 'Timestamp': Time at which consumer clicked on Ad or closed window
* 'Clicked on Ad': 0 or 1 indicated clicking on Ad

"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


def get_df(data_file):
    '''
    Read csv file of data and return a pandas dataframe.

    :param data_file: Local path to csv data file
    :return: 
    '''

    ad_data = pd.read_csv(data_file)

    # **Check the head of ad_data**
    ad_data.head()
    ad_data.info()
    ad_data.describe()

    return ad_data


def visualize_data(ad_data, target, features):
    '''
    Produce some Seaborn plots for data visualization of key features

    :param ad_data: dataframe 
    :param target: Identified target column (string)
    :param features: Identified feature columns (list)
    :return: 
    '''

    sns.distplot(ad_data[features[1]])

    sns.jointplot(ad_data[features[1]], ad_data[features[2]])

    sns.jointplot(ad_data[features[1]], ad_data[features[0]], kind='kde')

    sns.jointplot(ad_data[features[0]], ad_data[features[3]])

    sns.pairplot(ad_data, hue=target)

    plt.show()


def logistic_reg_train(ad_data, target, features):
    '''
    Split data into train and test sets, fit the data to a linear regression model.

    :param ad_data: dataframe
    :param target: Identified target column (string)
    :param features: Identified features columns (list)
    :return: 
    '''

    # Categorize features and target
    X = ad_data[features]
    y = ad_data[target]

    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    # Take a look at the head of your training set
    X_train.head()

    # Train and fit a logistic regression model on the training set.
    logm = LogisticRegression()
    logm.fit(X_train, y_train)

    return logm, X, y, X_test, y_test


def predictions(logm, X_test):
    '''
    Make predictions of test set using the logistic model developed.

    :param logm: Linear regression instance
    :param X_test: Test set of features
    :return: 
    '''

    predictions = logm.predict(X_test)

    return predictions


def model_evaluation(y_test, predictions):
    '''
    Evaluate the accuracy of the model employed.

    :param y_test: Test set of target data
    :param predictions: Predicted targets using model
    :return: 
    '''

    print(y_test)
    print(classification_report(y_test, predictions))


def main():
    '''

    :return: 
    '''

    # Bring in data csv as pandas dataframe
    chosen_file = 'advertising.csv'
    ad_data = get_df(chosen_file)

    # Look at snippet of dataframe
    print(ad_data.head())

    # Choose target and features from columns
    target = 'Clicked on Ad'
    features = ['Daily Time Spent on Site', 'Age', 'Area Income', 'Daily Internet Usage', 'Male']

    # Visualize data and correlations between features
    visualize_data(ad_data, target, features)

    # Split into train and test data and fit linear regression
    logm, X, y, X_test, y_test = logistic_reg_train(ad_data, target, features)

    # Predict results of test data
    prediction = predictions(logm, X_test)

    # Evaluate model
    model_evaluation(y_test, prediction)


print('\n This application takes a dataset and performs a logistic regression analysis on it. \n')
main()


