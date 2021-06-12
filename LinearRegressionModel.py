from sklearn import datasets,linear_model
import matplotlib.pyplot as plt 
import numpy as np
from sklearn.metrics import mean_squared_error
diabetes=datasets.load_diabetes()
# ['data', 'target', 'frame', 'DESCR', 'feature_names', 'data_filename', 'target_filename']
# print(diabetes.DESCR)
##Inbuilt 
##features  
# dia_x=diabetes.data[:, np.newaxis,2]
# dia_x_train=dia_x[:-20]
# dia_x_test=dia_x[:20]
##labels
# dia_y_train=diabetes.target[:-20]
# dia_y_test=diabetes.target[:20]
##manual
# features  
dia_x=np.array([[1],[2],[3]])
dia_x_train=dia_x
dia_x_test=dia_x
##labels
dia_y_train=np.array([[3],[2],[4]])
dia_y_test=np.array([3,2,4])
model=linear_model.LinearRegression()
model.fit(dia_x_train,dia_y_train)
PredictedTarget=model.predict(dia_x_test)
print("mean squared error is:",mean_squared_error(dia_y_test,PredictedTarget))
print("weight:",model.coef_)
print("Intercept:",model.intercept_)
plt.scatter(dia_x_test,dia_y_test)
plt.plot(dia_x_test,PredictedTarget)
plt.show()