#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


housing=pd.read_csv("housing_data.csv")


# In[3]:


housing.head()


# In[4]:


housing.head()


# In[5]:


housing.info()


# In[6]:


housing['CHAS'].value_counts()


# In[7]:


housing.describe()


# In[8]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[9]:


import matplotlib.pyplot as plt


# In[10]:


housing.hist(bins=50,figsize=(20,15))


# # Train-Test Splitting

# In[11]:


#code behind train_test_split
# def splittt(data,testratio):
#     np.random.seed(42)
#     shuffled=np.random.permutation(len(data))
#     test_set_size=int(len(data)*testratio)
#     test_indexs=shuffled[:test_set_size]
#     train_indexs=shuffled[test_set_size:]
#     return data.iloc[train_indexs],data.iloc[test_indexs]


# In[12]:


# train_set,test_set=splittt(housing,0.2)


# In[13]:


# print(test_set,train_set)


# In[14]:


from sklearn.model_selection import train_test_split


# In[15]:


train_set,test_set=train_test_split(housing,test_size=0.2,random_state=42)


# In[16]:


print(len(train_set),len(test_set))


# In[17]:


#since chas has less count of 1's which is 35 so it is possible that all 35 datapoints can go under test_set which has 102
#for  equa population distributuion we use stratified
# test_set["CHAS"].value_counts()


# In[18]:


from sklearn.model_selection import StratifiedShuffleSplit
split=StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)
for train_index,test_index in split.split(housing,housing['CHAS']):
    strat_train_set=housing.loc[train_index]
    strat_test_set=housing.loc[test_index]


# In[19]:


strat_test_set['CHAS'].value_counts()


# In[20]:


strat_train_set['CHAS'].value_counts()


# In[21]:


95/7


# In[22]:


376/28


# In[23]:


# ratio is equally distributed


# In[24]:


strat_train_set


# In[25]:


housing=strat_train_set.copy()


# # Correlation
# ### Value is between 0 and 1 and correlation of all features with respect to one we calculate that is if value of target feature increases then how other features are showing value that is by spearman coeff corelation if r==1 then its mostly the same feature if >0 <1 then +ve(value also will be increase/decreas) with respect of target feature
# ### if -ve then oppo result

# In[26]:


corr_matrix=housing.corr()


# In[27]:


corr_matrix['MEDV'].sort_values(ascending=False)


# In[28]:


from pandas.plotting import scatter_matrix
attributes=["MEDV","RM","AGE","LSTAT"]
scatter_matrix(housing[attributes],figsize=(12,8))


# In[29]:


housing.plot(kind="scatter",x="RM",y="MEDV",alpha=0.6)


# ## attribute combination

# In[30]:


housing['TAXRM']=housing["TAX"]/housing['RM']


# In[31]:


housing['TAXRM']


# In[32]:


housing.head()


# In[33]:


corr_matrix=housing.corr()
corr_matrix['MEDV'].sort_values(ascending=False)


# In[34]:


housing.plot(kind="scatter",x="TAXRM",y="MEDV",alpha=0.6)


# # missing attributes

# In[35]:


housing=strat_train_set.drop("MEDV",axis=1)
housing_labels=strat_train_set['MEDV'].copy()


# In[36]:


# to care of missing attributes ,u have 3 options
#     1.Get rid of DataPoints
#     2.Get rid of whole attributes
#     3.Set the value to 0 ,mean or median.


# In[37]:


a=housing.dropna(subset=["RM"])#OPTION 1
a.shape


# In[38]:


housing.drop('RM',axis=1).shape#option 2
#nNote that original housing dataframe willl remain unchange


# In[39]:


median=housing['RM'].median()


# In[40]:


housing['RM'].fillna(median)#Option 3


# In[41]:


housing.shape


# In[42]:


housing.describe()#before we started imputing


# # Imputer
# ### Imputer changes all na values to mean median to original data if you do that indirectly changes may not occur to original data

# In[43]:



from sklearn.impute import SimpleImputer
imputer=SimpleImputer(strategy="median")
imputer.fit(housing)


# In[44]:


imputer.statistics_


# In[45]:


x=imputer.transform(housing)


# In[46]:


housing_tr=pd.DataFrame(x,columns=housing.columns)


# In[47]:


housing_tr


# In[48]:


housing_tr.describe()


# # Scikikt-learn design

# 3 objects
# 1.Estimator-e.g imputer
# 2.Transformator-e.g.transfrom method takes input and returns based on the learning from fit()
# 3.Predictors..

# # Feature Scaling

# Primarily two types of feature scaling method:
# Machine learning algorithm just sees number — if there is a vast difference in the range say few ranging in thousands and few ranging in the tens, and it makes the underlying assumption that higher ranging numbers have superiority of some sort. So these more significant number starts playing a more decisive role while training the model.
# 1)Min Max scaling(Normalization)
#     (value-min)/(min-max)--values from 0 to 1
#     MinMaxScaler in sckitlearn
# 2)standardlization
#     (value-mean)/std 
#     standardscalar in scikitlearn

# # Creating pipeline

# In[49]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
my_pipeline=Pipeline([
    ('Imputer',SimpleImputer(strategy="median")),
    ('std_scalar',StandardScaler()),
])


# In[50]:


housing_num_tr=my_pipeline.fit_transform(housing)
housing_num_tr


# In[51]:


housing_num_tr.shape


# # Selecting desired model

# In[52]:


from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
# model=LinearRegression()
model=RandomForestRegressor()
# model=DecisionTreeRegressor()
model.fit(housing_num_tr,housing_labels)


# In[53]:


some_data=housing.iloc[:5]
somedata_labels=housing_labels.iloc[:5]


# In[68]:


prepared_data=my_pipeline.transform(some_data)
prepared_data[0]


# In[55]:


model.predict(prepared_data)


# In[56]:


list(somedata_labels)


# In[57]:


from sklearn.metrics import mean_squared_error
housing_predictions=model.predict(housing_num_tr)
mse=mean_squared_error(housing_labels,housing_predictions)
mse_sq=np.sqrt(mse)


# In[58]:


mse


# In[59]:


mse_sq


# ## Using better evaluating technique-Cross validation

# In[60]:


from sklearn.model_selection import cross_val_score
scores=cross_val_score(model,housing_num_tr,housing_labels,scoring="neg_mean_squared_error",cv=10)
rmse_scores=np.sqrt(-scores)
rmse_scores


# In[61]:


def print_scores(scores):
    print('Scores are',scores)
    print('Mean: ',scores.mean())
    print("Std :",scores.std())
    print("var :",scores.var())
    


# In[62]:


print_scores(rmse_scores)


# # Model Saving

# In[63]:


from joblib import dump,load
dump(model,'HousingPrice.joblib')


# # Testing

# In[64]:


x_test=strat_test_set.drop('MEDV',axis=1)
y_test=strat_test_set['MEDV'].copy()
x_test_prepares=my_pipeline.transform(x_test)
finalpredictions=model.predict(x_test_prepares)
final_mse=mean_squared_error(y_test,finalpredictions)
final_rmse=np.sqrt(final_mse)


# In[65]:


final_rmse


# In[66]:


np.array(y_test)


# In[67]:


np.array(finalpredictions)


# In[ ]:




