# $Id: KNN_Confidential_Data_Analysis.py 8163 2018-03-15 05:09:30Z milde $
# Author: Megan McClarty

"""
K Nearest Neighbors Analysis on Confidential Data

This generalized Python file is modified from a Jupyter notebook created to complete the K Nearest Neighbors Project 
portion of Udemy course Python for Data Science and Machine Learning.


"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


def get_df(data_file):
    '''
    Read csv file of data and return a pandas dataframe.

    :param data_file: Local path to csv data file
    :return: 
    '''

    knn_df = pd.read_csv(data_file)
    print(knn_df.head())

    return knn_df


def visualize_data(knn_df, target, features):
    '''
    Produce some Seaborn plots for data visualization of key features

    :param knn_df: dataframe 
    :param target: Identified target column (string)
    :param features: Identified feature columns (list)
    :return: 
    '''

    sns.pairplot(knn_df, hue=target)
    plt.show()


def scale_data(knn_df, target):
    '''

    :return: 
    '''

    scaler = StandardScaler()
    scaler.fit(knn_df.drop(target, axis=1))
    scaled_features = scaler.transform(knn_df.drop(target, axis=1))
    sc_feat = pd.DataFrame(scaled_features, columns=knn_df.columns[:-1])
    sc_feat.head()

    return sc_feat


def knn_train(sc_feat, knn_df, target):
    """
    
    :param sc_feat: df
    :param knn_df: df
    :param target: column name
    :return: 
    """

    N = 16

    X_train, X_test, y_train, y_test = train_test_split(sc_feat,knn_df[target],test_size=0.3)
    knn_instance = KNeighborsClassifier(n_neighbors=N)
    knn_instance.fit(X_train, y_train)

    return knn_instance, X_train, X_test, y_train, y_test


def predictions(knn_instance, X_test):
    '''
    Make predictions of test set using the logistic model developed.

    :param knn_instance: Linear regression instance
    :param X_test: Test set of features
    :return: 
    '''

    predictions = knn_instance.predict(X_test)

    return predictions


def model_evaluation(X_train, X_test, y_train, y_test, prediction):
    '''
    Evaluate model.
    
    :param X_train: 
    :param X_test: 
    :param y_train: 
    :param y_test: 
    :param prediction: 
    :return: 
    '''

    conf = confusion_matrix(y_test, prediction)
    print(conf)
    print(classification_report(y_test, prediction))

    error_rate = []

    for i in range(1, 40):
        knn_instance = KNeighborsClassifier(n_neighbors=i)
        knn_instance.fit(X_train, y_train)
        pred_i = knn_instance.predict(X_test)
        error_rate.append(np.mean(pred_i != y_test))

    plt.figure()
    plt.plot(range(1, 40), error_rate)


def main():
    '''

    :return: 
    '''

    # Bring in data csv as pandas dataframe
    chosen_file = 'KNN_Project_Data'
    knn_df = get_df(chosen_file)

    # Choose target and features from columns
    target = 'TARGET CLASS'
    features = list(knn_df.columns[:-1])

    # Visualize data
    visualize_data(knn_df, target, features)

    # Scale data
    sc_feat = scale_data(knn_df, target)

    # Split into train and test data and fit KNN model
    knn_instance, X_train, X_test, y_train, y_test = knn_train(sc_feat, knn_df, target)

    # Predict results of test data
    prediction = predictions(knn_instance, X_test)

    # Evaluate model
    model_evaluation(X_train, X_test, y_train, y_test, prediction)


print('\n This application takes a confidential (unlabeled) dataset and performs a K-Nearest Neighbors analysis'
      ' on it. \n')
main()




