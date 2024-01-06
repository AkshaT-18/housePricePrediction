import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

df = pd.read_csv('Housing.csv')


#Converting categorical Columns to Numeric
cc=['mainroad','guestroom','basement','hotwaterheating','airconditioning','prefarea']
for i in cc:
  df[i].mask(df[i]=='yes',1,inplace = True)
  df[i].mask(df[i]=='no',0,inplace = True)
#one-hot encoding
df = pd.get_dummies(df,columns=['furnishingstatus'], drop_first=True)


#removing outliers
df1 = df.copy()
Q1 = df1['area'].quantile(0.25)
Q3 = df1['area'].quantile(0.75)
IQR = Q3 - Q1
df1 = df1[df1['area'] <= (Q3+(1.5*IQR))]
df1 = df1[df1['area'] >= (Q1-(1.5*IQR))]
df1 = df1.reset_index(drop=True)
df = df1.copy()


X = df.drop(['price'],axis=1)
Y = df['price']
Train_X, Test_X, Train_Y, Test_Y = train_test_split(X, Y, train_size=0.8, test_size=0.2, random_state=100)
Train_X.reset_index(drop=True,inplace=True)

#Feature Scaling (Standardization)

std = StandardScaler()

# Standardardization on Training set
Train_X_std = std.fit_transform(Train_X)
Train_X_std = pd.DataFrame(Train_X_std, columns=X.columns)

# Standardardization on Testing set
Test_X_std = std.transform(Test_X)
Test_X_std = pd.DataFrame(Test_X_std, columns=X.columns)

MLR = LinearRegression().fit(Train_X_std,Train_Y)
y_pred = MLR.predict(Test_X_std)

pickle.dump(MLR,open('model.pkl','wb'))

model = pickle.load(open('model.pkl','rb'))