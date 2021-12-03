import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score,accuracy_score,confusion_matrix,recall_score
import pickle

df = pd.read_csv('data_files/student-mat.csv',sep=';')
df = df[['G1','G2','G3','studytime','failures','absences']]

predict = df['G3'].values
X = df[['G1','G2','studytime','failures','absences']].values

y = predict

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.1)

"""
linear = LinearRegression()
linear.fit(X_train,y_train)


with open("student_linear_model.pickle","wb") as f:
    pickle.dump(linear,f)
"""

pickle_in = open("student_linear_model.pickle","rb")
linear = pickle.load(pickle_in)
linear.fit(X_train,y_train)

y_pred = linear.predict(X_test)
print(linear.score(X_test,y_test))

for x in range(len(y_pred)):
    print("y_pred: ",y_pred[x],"X_test:",X_test[x],"y_test:",y_test[x])

"""
# Finding the best model
accuraxix = []
for i in range(5):

    linear = LinearRegression()
    linear.fit(X_train,y_train)

    y_pred = linear.predict(X_test)
    for x in range(len(y_pred)):
        print(y_pred[x],X_test[x],y_test[x])
    accuracy = {
                "Accuracy: ":linear.score(X_test,y_test),
                "Coef: ":linear.coef_,
                "Intercept: ":linear.intercept_
            }
    accuraxix.append(accuracy)
    print(accuraxix)
"""

#{{notes}}
#coefitient is the gradient
#in 2d there is one gradient 
#but in 6d there are 5 gradients
#the intercept is where all of the lines cross the y-axis
#{{notes}}

"""

"""