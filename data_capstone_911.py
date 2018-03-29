# $Id: data_capstone_911.py 8163 2018-03-29 15:46:00Z milde $
# Author: Megan McClarty <m_mcclarty@outlook.com>

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

'''
This is an export of a Jupyter notebook file.

This project is taken from the curriculum of Udemy course "Python for Data Science and Machine Learning".

The '911.csv' dataset is bundled with this file.  It is a Kaggle dataset that includes details of calls made to the
911 emergency line.
'''

df = pd.read_csv('911.csv')
df.info()
print(df.head())

print('\n Sort the dataset by count of 911 calls by zipcode')
df_by_zip = df.groupby(['zip']).size()
series = df_by_zip.sort_values(ascending=False)
series.head()

print('\n Sort the dataset by count of 911 calls by township')
df_by_twp = df.groupby(['twp']).size()
series = df_by_twp.sort_values(ascending=False)
series.head()

print('\n Split the title column to create a new column which extracts only the categorization of that call '
      '(EMS, Traffic, Fire)')
df['Reason'] = df['title'].apply(lambda string : string.split(':')[0])
print(df.head())

print('\n Find and visualize the most common reason for a 911 call')
df.groupby(['Reason']).count()
sns.countplot(df['Reason'])
plt.show()

print('\n Convert the timestamp column to datetime (from string)')
df['timeStamp'] = pd.to_datetime(df['timeStamp'])

print('\n Extract the day of the week from the timeStamp and map to a string representation of the day of the week')
dmap = {0: 'Mon', 1:'Tue', 2: 'Wed', 3: 'Thu', 4: 'Fri', 5: 'Sat', 6: 'Sun'}
df['Day of Week'] = (df['timeStamp'].apply(lambda time : time.dayofweek)).map(dmap)
sns.countplot(df['Day of Week'], hue=df['Reason'])
plt.show()

print('\n Repeat for months; but do not map, since you will later plot by numeric representation of the month')
#mmap = {1: 'January', 2: 'February', 3: 'March', 4: 'April', 5: 'May', 6 : 'June', 7 : 'July', 8 : 'August', 9 : 'September', 10 : 'October', 11 : 'November', 12 : 'December'}
df['Month of Year'] = df['timeStamp'].apply(lambda time : time.month)
sns.countplot(df['Month of Year'], hue=df['Reason'])
plt.show()

print('\n Build a line plot to visualize trend during months where data is missing, then find linear fit')
byMonth = df.groupby('Month of Year').count()
plt.plot(byMonth.index,byMonth['lat'])

byMonth['month_index'] = byMonth.index
sns.lmplot('month_index', 'lat', byMonth)
plt.show()

print('\n Repeat this analysis but for the date')
df['Date'] = df['timeStamp'].apply(lambda time : time.date())
df.head()
byDate = df.groupby(df['Date']).count()
plt.plot(byDate.index, byDate['lat'])
plt.show()

print('\n Plot all trends separately by Reason')
reasons = ['Fire', 'EMS', 'Traffic']
for i in reasons:
    only_reasons = df.loc[df['Reason'] == i]
    byDate_reasons = only_reasons.groupby(only_reasons['Date']).count()
    plt.plot(byDate_reasons.index, byDate_reasons['lat'])
    plt.title('Calls by Reason')
    plt.legend(reasons)
    plt.show()
