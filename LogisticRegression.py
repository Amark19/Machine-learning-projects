"""about logistic regression
It is an classfier which uses linear regression to find probabilty by giving predcited label value as input to it
It uses signoid function(1/(1+e^-x))(y co-ordinate are always between 0 to 1) to convert label value to either 0 or 1 whose .
If the probability of some label is >0.5 then it is 1 or its 0.
"""
#code
from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.linear_model import LogisticRegression
iris=datasets.load_iris()
# print(list(iris.keys()))
# print(iris['feature_names'])
# print(iris['target_names'])

x=iris['data'][:,3:]
y=(iris['target']==2).astype(np.int)#to make it classifier whether it is verginica or not
# print(y)
#training
clf=LogisticRegression()
clf.fit(x,y)
#test
exam=clf.predict([[0.7]])
# print(exam)
#using matplotlib
x_new=np.linspace(0,3,1000).reshape(-1,1)
y_prob=clf.predict_proba(x_new)
# print(y_prob)
# plt.plot(x_new,y_prob[:,1],"g-",iris['target']==2)
plt.plot(x_new,y_prob[:,1],"g-",label="virginica")
plt.show()