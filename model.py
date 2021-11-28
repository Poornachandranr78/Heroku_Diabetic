import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
#data loading by using pandas
data=pd.read_csv("C:\\Users\\poorn\\diabetes.csv")
#Data frame column rename
data=data.rename(columns={"DiabetesPedigreeFunction":"DPF"})
print(data.head())
#the dataset is not general so we have to make it as general data set by using  the following method
data['BMI'] = data['BMI'].replace(0,data['BMI'].mean())
data['BloodPressure'] = data['BloodPressure'].replace(0,data['BloodPressure'].mean())
data['Glucose'] = data['Glucose'].replace(0,data['Glucose'].mean())
data['Insulin'] = data['Insulin'].replace(0,data['Insulin'].mean())
data['SkinThickness'] = data['SkinThickness'].replace(0,data['SkinThickness'].mean())
#we have to remove the outfilers
q = data['Pregnancies'].quantile(0.98)
# we are removing the top 2% data from the Pregnancies column
data_cleaned = data[data['Pregnancies']<q]
q = data_cleaned['BMI'].quantile(0.99)
# we are removing the top 1% data from the BMI column
data_cleaned  = data_cleaned[data_cleaned['BMI']<q]
q = data_cleaned['SkinThickness'].quantile(0.99)
# we are removing the top 1% data from the SkinThickness column
data_cleaned  = data_cleaned[data_cleaned['SkinThickness']<q]
q = data_cleaned['Insulin'].quantile(0.95)
# we are removing the top 5% data from the Insulin column
data_cleaned  = data_cleaned[data_cleaned['Insulin']<q]
q = data_cleaned['DPF'].quantile(0.99)
# we are removing the top 1% data from the DiabetesPedigreeFunction column
data_cleaned  = data_cleaned[data_cleaned['DPF']<q]
q = data_cleaned['Age'].quantile(0.99)
# we are removing the top 1% data from the Age column
data_cleaned  = data_cleaned[data_cleaned['Age']<q]
#we have to split the X and Y column
x=data.drop(columns="Outcome")
y=data["Outcome"]
#we have to split the test into two types
train_x,test_x,train_y,test_y=train_test_split(x,y,random_state=0,test_size=.20)
#model development
clf=RandomForestClassifier(n_estimators=20)
clf.fit(train_x,train_y)
#trainnig data score
s=clf.score(train_x,train_y)
print("the train data score is",s)
#pickle constuct
filename = 'diabetes-prediction-rfc-models.pkl'
pickle.dump(clf, open(filename, 'wb'))
