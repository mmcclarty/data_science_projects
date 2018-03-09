# $Id: Advertising_Logistic_Regression.py 8163 2018-03-08 05:09:30Z milde $
# Author: Megan McClarty

"""
Logistic Regression Project 

In this project we will be working with a fake advertising data set, indicating whether or not a particular internet user clicked on an Advertisement. We will try to create a model that will predict whether or not they will click on an ad based off the features of that user.

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


def get_df(data_file):
    '''
    Read csv file of data and return a pandas dataframe.

    :param data_file: Local path to csv data file
    :return: 
    '''

    # ## Get the Data
    # **Read in the advertising.csv file and set it to a data frame called ad_data.**

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

    sns.distplot(ad_data['Age'])

    # **Create a jointplot showing Area Income versus Age.**

    sns.jointplot(ad_data['Age'], ad_data['Area Income'])

    # **Create a jointplot showing the kde distributions of Daily Time spent on site vs. Age.**

    sns.jointplot(ad_data['Age'], ad_data['Daily Time Spent on Site'], kind='kde')

    # ** Create a jointplot of 'Daily Time Spent on Site' vs. 'Daily Internet Usage'**

    sns.jointplot(ad_data['Daily Time Spent on Site'], ad_data['Daily Internet Usage'])

    # ** Finally, create a pairplot with the hue defined by the 'Clicked on Ad' column feature.**

    sns.pairplot(ad_data, hue='Clicked on Ad')

    plt.show()


def logistic_reg_train(ad_data, target, features):
    '''
    Split data into train and test sets, fit the data to a linear regression model.

    :param ad_data: dataframe
    :param target: Identified target column (string)
    :param features: Identified features columns (list)
    :return: 
    '''

    # # Logistic Regression
    #
    # Now it's time to do a train test split, and train our model!
    #
    # You'll have the freedom here to choose columns that you want to train on!

    # ** Split the data into training set and testing set using train_test_split**
    from sklearn.model_selection import train_test_split
    X = ad_data[features]
    y = ad_data[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    X_train.head()

    # ** Train and fit a logistic regression model on the training set.**
    from sklearn.linear_model import LogisticRegression
    logm = LogisticRegression()
    logm.fit(X_train, y_train)

    return logm, X, y, X_test, y_test


def predictions(logm, X_test, y_test):
    '''
    Make predictions of test set using the linear model developed.

    :param logm: Linear regression instance
    :param X_test: Test set of features
    :param y_test: Test set of target
    :return: 
    '''

    # ## Predictions and Evaluations
    # ** Now predict values for the testing data.**

    # In[44]:

    predictions = logm.predict(X_test)
    print(predictions)
    return predictions


def model_evaluation(y_test, predictions):
    '''
    Evaluate the accuracy of the model employed.

    :param y_test: Test set of target data
    :param predictions: Predicted targets using model
    :return: 
    '''

    from sklearn.metrics import classification_report
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
    prediction = predictions(logm, X_test, y_test)

    # Evaluate model
    model_evaluation(y_test, prediction)


print('\n This application takes a dataset and performs a logistic regression analysis on it. \n')
main()


