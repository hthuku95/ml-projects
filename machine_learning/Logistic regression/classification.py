'''
classification is part of supervised learning.
there are two categories in supervised learning, classification and regression.
Regression works well with numerical values while classification works well with boolean values.

'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer

df = pd.read_csv("data_files/titanic.csv")
print(df.head())

#building a logistical regression model
df['male'] = df['Sex'] == 'male'

#Target || Data to be predictd
y = df['Survived'].values

#Feature data || Training data
X = df[['Pclass','male','Age','Siblings/Spouses','Parents/Children','Fare']].values

print(X)
print(y)

#calling the logistic regression function
model = LogisticRegression()
#creating the best fit line
model.fit(X,y)

print(model.coef_,model.intercept_)
print(model.predict(X[:20]))
print(y[:20])

#calculating the accuracy of the model
y_pred = model.predict(X)

model_accuracy = (y == y_pred).sum() / y.shape[0]
print(model_accuracy)
print(model.score(X,y))

cancer_data = load_breast_cancer()
print(cancer_data.keys())
print(cancer_data['DESCR'])

data = pd.DataFrame(cancer_data['data'],columns = cancer_data['feature_names'])
print(data.head())