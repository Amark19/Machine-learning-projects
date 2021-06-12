from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
#Loading datasets
iris=datasets.load_iris()
# printing description and features
# print(iris.DESCR)
features=iris.data
labels=iris.target
# print(features[0],labels[0])
#Training Classifier
clf=KNeighborsClassifier()
clf.fit(features,labels)
preds=clf.predict([[3,3,3,3]])
print(preds)