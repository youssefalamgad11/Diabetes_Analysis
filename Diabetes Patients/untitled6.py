# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 01:52:24 2024

@author: youss
"""


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix , classification_report ,accuracy_score

df=pd.read_csv('diabetes.csv')
df.head()

df.info()
df.describe()

df.duplicated().sum()

count_diabetes=df['Outcome'].value_counts()
count_diabetes

plt.figure(figsize=(12,8))
sns.barplot(x=count_diabetes.index,y=count_diabetes.values,palette='flare')
plt.title('count_diabetes')
plt.xlabel('diabetes')
plt.ylabel('count')
plt.show()

age=sns.FacetGrid(df,col='Outcome')
age.map(plt.hist,'Age')

DiabetesPedigreeFunction=sns.FacetGrid(df,col='Outcome')
DiabetesPedigreeFunction.map(plt.hist,'DiabetesPedigreeFunction')

BMI=sns.FacetGrid(df,col='Outcome')
BMI.map(plt.hist,'BMI')

Insulin=sns.FacetGrid(df,col='Outcome')
Insulin.map(plt.hist,'Insulin')

SkinThickness=sns.FacetGrid(df,col='Outcome')
SkinThickness.map(plt.hist,'SkinThickness')

def plot_scatter(df, cols, col_y = 'Age'):
    for col in cols:
        fig = plt.figure(figsize=(12,6)) # define plot area
        ax = fig.gca() # define axis   
        df.plot.scatter(x = col, y = col_y, ax = ax)
        ax.set_title('Scatter plot of ' + col_y + ' vs. ' + col) # Give the plot a main title
        ax.set_xlabel(col) # Set text for the x axis
        ax.set_ylabel(col_y
                     )# Set text for y axis
        plt.show()

num_cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'DiabetesPedigreeFunction','BMI','Insulin']
plot_scatter(df, num_cols) 

plt.figure(figsize=(10,6))
plt.boxplot([df['Age'], df['BMI'], df['BloodPressure'], df['Glucose']], vert=False)
plt.yticks([1, 2, 3, 4], ['Age', 'BMI', 'BloodPressure', 'Glucose'])
plt.xlabel('Value')
plt.title("Box Plot")

X=df.iloc[:,:-1]
y=df['Outcome']
y=y.values.reshape(-1,1)

scaler=MinMaxScaler()
X=scaler.fit_transform(X)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,shuffle=True,random_state=42)

lg=LogisticRegression()
lg.fit(X_train,y_train)

lg.score(X_train,y_train)

y_pred=lg.predict(X_test)

print(accuracy_score(y_test,y_pred))

from sklearn.metrics import ConfusionMatrixDisplay
ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
print(confusion_matrix(y_test, y_pred))

print(classification_report(y_test, y_pred))

