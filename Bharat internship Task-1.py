#!/usr/bin/env python
# coding: utf-8

# # Bharat Intern Internship

# # Task 1 : Titanic Classification

# By krishna gollapalli

# # About the Dataset

# On April 15, 1912, during her maiden voyage, the widely considered “unsinkable” RMS Titanic sank after colliding with an iceberg. Unfortunately, there weren’t enough lifeboats for everyone on board, resulting in the death of 1502 out of 2224 passengers and crew.
# 
# While there was some element of luck involved in surviving, it seems some groups of people were more likely to survive than others.
# 
# In this challenge, we ask you to build a predictive model that answers the question: “what sorts of people were more likely to survive?” using passenger data (ie name, age, gender, socio-economic class, etc).
# 
# Objective:
# 
# 1.Understand the Dataset & cleanup (if required).
# 
# 2.Build a strong classification model to predict whether the passenger survives or not.
# 
# 3.Also fine-tune the hyperparameters & compare the evaluation metrics of various classification algorithms.
# 
# 

# # Link to the Dataset :

# # https://www.kaggle.com/datasets/yasserh/titanic-dataset

# # Data Preparation

# In[1]:


# Importing Libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')


# In[3]:


# Loading the Dataset
titanic = pd.read_csv('Titanic-Dataset.csv')
titanic


# In[4]:


# Reading first 5 rows
titanic.head()


# In[5]:


# Reading last 5 rows
titanic.tail()


# In[6]:


# Showing no. of rows and columns of dataset
titanic.shape


# In[7]:


# checking for columns
titanic.columns


# Information about Columns
# 
# 1.PassengerId: unique id number to each passenger
# 
# 2.Survived: passenger survive(1) or died(0)
# 
# 3.Pclass: passenger class
# 
# 4.Name: name
# 
# 5.Sex: gender of passenger
# 
# 6.Age: age of passenger
# 
# 7.SibSp: number of siblings/spouses
# 
# 8.Parch: number of parents/children
# 
# 9.Ticket: ticket number
# 
# 10.Fare: amount of money spent on ticket
# 
# 11.Cabin: cabin category
# 
# 12.Embarked: port where passenger embarked (C = Cherbourg, Q = Queenstown, S = Southampton
# 

# # Data Preprocessing and Data Cleaning

# In[8]:


# Checking for data types
titanic.dtypes


# # Handling Duplicated Values

# In[9]:


# checking for duplicated values
titanic.duplicated().sum()


# # Null Values Treatment

# In[10]:


# checking for null values
nv = titanic.isna().sum().sort_values(ascending=False)
nv = nv[nv>0]
nv


# In[11]:


# Cheecking what percentage column contain missing values
titanic.isnull().sum().sort_values(ascending=False)*100/len(titanic)


# In[12]:


# Since Cabin Column has more than 75 % null values .So , we will drop this column
titanic.drop(columns = 'Cabin', axis = 1, inplace = True)
titanic.columns


# In[15]:


# Filling Null Values in Age column with mean values of age column
titanic['Age'].fillna(titanic['Age'].mean(),inplace=True)

# filling null values in Embarked Column with mode values of embarked column
titanic['Embarked'].fillna(titanic['Embarked'].mode()[0],inplace=True)


# In[16]:


# checking for null values
titanic.isna().sum()


# # Checking for unique values

# In[18]:


# Finding no. of unique values in each column of dataset
titanic[['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp',
       'Parch', 'Ticket', 'Fare', 'Embarked']].nunique().sort_values()


# # Showing unique values of different columns

# In[19]:


titanic['Survived'].unique()


# In[20]:


titanic['Sex'].unique()


# In[21]:


titanic['Pclass'].unique()


# In[22]:


titanic['SibSp'].unique()


# In[23]:


titanic['Parch'].unique()


# In[24]:


titanic['Embarked'].unique()


# # Dropping Some Unnecessary Columns

# There are 3 columns i.e.. 'PassengerId' , 'Name' , 'Ticket' are unnecessary columns which have no use in data modelling . So, we will drop these 3 columns

# In[25]:


titanic.drop(columns=['PassengerId','Name','Ticket'],axis=1,inplace=True)
titanic.columns


# In[26]:


# Showing inforamation about the dataset
titanic.info()


# In[27]:


# showing info about numerical columns
titanic.describe()


# In[28]:


# showing info about categorical columns
titanic.describe(include='O')


# # Data Visualization

# 1.Sex Column

# In[29]:


d1 = titanic['Sex'].value_counts()
d1


# In[30]:


# Plotting Count plot for sex column
sns.countplot(x=titanic['Sex'])
plt.show()


# In[31]:


# Plotting Percantage Distribution of Sex Column
plt.figure(figsize=(5,5))
plt.pie(d1.values,labels=d1.index,autopct='%.2f%%')
plt.legend()
plt.show()


# In[32]:


# Showing Distribution of Sex Column Survived Wise
sns.countplot(x=titanic['Sex'],hue=titanic['Survived']) # In Sex (0 represents female and 1 represents male)
plt.show()


# This plot clearly shows male died more than females and females survived more than male percentage.

# In[33]:


# Showing Distribution of Embarked Sex wise
sns.countplot(x=titanic['Embarked'],hue=titanic['Sex'])
plt.show()


# 2.Pclass Column

# In[34]:


# Plotting CountPlot for Pclass Column
sns.countplot(x=titanic['Pclass'])
plt.show()


# In[35]:


# Showing Distribution of Pclass Sex wise
sns.countplot(x=titanic['Pclass'],hue=titanic['Sex'])
plt.show()


# In[36]:


# Age Distribution
sns.kdeplot(x=titanic['Age'])
plt.show()


# From this plot it came to know that most of the people lie between 20-40 age group.
# 
# Analysing Target Variable
# 
# Survived Column

# In[37]:


# Plotting CountPlot for Survived Column
print(titanic['Survived'].value_counts())
sns.countplot(x=titanic['Survived'])
plt.show()


# This plot Clearly shows most people are died

# In[38]:


# Showing Distribution of Parch Survived Wise
sns.countplot(x=titanic['Parch'],hue=titanic['Survived'])
plt.show()


# In[39]:


# Showing Distribution of SibSp Survived Wise
sns.countplot(x=titanic['SibSp'],hue=titanic['Survived'])
plt.show()


# In[40]:


# Showing Distribution of Embarked Survived wise
sns.countplot(x=titanic['Embarked'],hue=titanic['Survived'])
plt.show()


# In[41]:


# Showinf Distribution of Age Survived Wise
sns.kdeplot(x=titanic['Age'],hue=titanic['Survived'])
plt.show()


# This Plot showing most people of age group of 20-40 are died

# In[42]:


# Plotting Histplot for Dataset
titanic.hist(figsize=(10,10))
plt.show()


# In[43]:


# Showing Correlation Plot
sns.heatmap(titanic.corr(),annot=True,cmap='coolwarm')
plt.show()


# This Plot is clearly showing
# 
# 1.Strong Positive Correlation between SibSp and Parch
# 
# 2.Strong Negative Correlation between Pclass and Fare

# In[44]:


# Plotting pairplot
sns.pairplot(titanic)
plt.show()


# # Checking the target variable

# In[45]:


titanic['Survived'].value_counts()


# In[46]:


sns.countplot(x=titanic['Survived'])
plt.show()


# # Label Encoding

# In[47]:


from sklearn.preprocessing import LabelEncoder
# Create an instance of LabelEncoder
le = LabelEncoder()

# Apply label encoding to each categorical column
for column in ['Sex','Embarked']:
    titanic[column] = le.fit_transform(titanic[column])

titanic.head()

# Sex Column

# 0 represents female
# 1 represents Male

# Embarked Column

# 0 represents C
# 1 represents Q
# 2 represents S


# # Data Modelling

# In[48]:


# importing libraries

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score


# # Selecting the independent and dependent Features

# In[49]:


cols = ['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']
x = titanic[cols]
y = titanic['Survived']
print(x.shape)
print(y.shape)
print(type(x))  # DataFrame
print(type(y))  # Series


# In[50]:


x.head()


# In[51]:


y.head()


# # Train_Test_Split

# In[52]:


print(891*0.10)


# In[53]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.10,random_state=1)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


# # Creating Functions to compute Confusion Matrix, Classification Report and to generate Training and the Testing Score(Accuracy)

# In[54]:


def cls_eval(ytest,ypred):
    cm = confusion_matrix(ytest,ypred)
    print('Confusion Matrix\n',cm)
    print('Classification Report\n',classification_report(ytest,ypred))

def mscore(model):
    print('Training Score',model.score(x_train,y_train))  # Training Accuracy
    print('Testing Score',model.score(x_test,y_test))     # Testing Accuracy


# 1.Logistic Regression

# In[55]:


# Building the logistic Regression Model
lr = LogisticRegression(max_iter=1000,solver='liblinear')
lr.fit(x_train,y_train)


# In[56]:


# Computing Training and Testing score
mscore(lr)


# In[57]:


# Generating Prediction
ypred_lr = lr.predict(x_test)
print(ypred_lr)


# In[58]:


# Evaluate the model - confusion matrix, classification Report, Accuracy score
cls_eval(y_test,ypred_lr)
acc_lr = accuracy_score(y_test,ypred_lr)
print('Accuracy Score',acc_lr)


# In[59]:


# Building the knnClassifier Model
knn=KNeighborsClassifier(n_neighbors=8)
knn.fit(x_train,y_train)


# In[60]:


# Computing Training and Testing score
mscore(knn)


# In[61]:


# Generating Prediction
ypred_knn = knn.predict(x_test)
print(ypred_knn)


# In[62]:


# Evaluate the model - confusion matrix, classification Report, Accuracy score
cls_eval(y_test,ypred_knn)
acc_knn = accuracy_score(y_test,ypred_knn)
print('Accuracy Score',acc_knn)


# In[63]:


# Building Support Vector Classifier Model
svc = SVC(C=1.0)
svc.fit(x_train, y_train)


# In[64]:


# Computing Training and Testing score
mscore(svc)


# In[65]:


# Generating Prediction
ypred_svc = svc.predict(x_test)
print(ypred_svc)


# In[66]:


# Evaluate the model - confusion matrix, classification Report, Accuracy score
cls_eval(y_test,ypred_svc)
acc_svc = accuracy_score(y_test,ypred_svc)
print('Accuracy Score',acc_svc)


# In[67]:


# Building the RandomForest Classifier Model
rfc=RandomForestClassifier(n_estimators=80,criterion='entropy',min_samples_split=5,max_depth=10)
rfc.fit(x_train,y_train)


# In[68]:


# Computing Training and Testing score
mscore(rfc)


# In[69]:


# Generating Prediction
ypred_rfc = rfc.predict(x_test)
print(ypred_rfc)


# In[70]:


# Evaluate the model - confusion matrix, classification Report, Accuracy score
cls_eval(y_test,ypred_rfc)
acc_rfc = accuracy_score(y_test,ypred_rfc)
print('Accuracy Score',acc_rfc)


# In[71]:


# Building the DecisionTree Classifier Model
dt = DecisionTreeClassifier(max_depth=5,criterion='entropy',min_samples_split=10)
dt.fit(x_train, y_train)


# In[72]:


# Computing Training and Testing score
mscore(dt)


# In[73]:


# Generating Prediction
ypred_dt = dt.predict(x_test)
print(ypred_dt)


# In[74]:


# Evaluate the model - confusion matrix, classification Report, Accuracy score
cls_eval(y_test,ypred_dt)
acc_dt = accuracy_score(y_test,ypred_dt)
print('Accuracy Score',acc_dt)


# In[75]:


# Builing the Adaboost model
ada_boost  = AdaBoostClassifier(n_estimators=80)
ada_boost.fit(x_train,y_train)


# In[76]:


# Computing the Training and Testing Score
mscore(ada_boost)


# In[77]:


# Generating the predictions
ypred_ada_boost = ada_boost.predict(x_test)


# In[78]:


# Evaluate the model - confusion matrix, classification Report, Accuracy Score
cls_eval(y_test,ypred_ada_boost)
acc_adab = accuracy_score(y_test,ypred_ada_boost)
print('Accuracy Score',acc_adab)


# In[79]:


models = pd.DataFrame({
    'Model': ['Logistic Regression','knn','SVC','Random Forest Classifier','Decision Tree Classifier','Ada Boost Classifier'],
    'Score': [acc_lr,acc_knn,acc_svc,acc_rfc,acc_dt,acc_adab]})

models.sort_values(by = 'Score', ascending = False)


# In[80]:


colors = ["blue", "green", "red", "yellow","orange","purple"]

sns.set_style("whitegrid")
plt.figure(figsize=(15,5))
plt.ylabel("Accuracy %")
plt.xlabel("Algorithms")
sns.barplot(x=models['Model'],y=models['Score'], palette=colors )
plt.show()


# DecisionTree Classifier Model got the Highest Accuracy

# In[ ]:




