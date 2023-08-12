#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import all the required library and installsklearna and 
import numpy as np
import pandas as pd 
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from numpy import genfromtxt
from sklearn.model_selection import train_test_split
import random

from sklearn.impute import SimpleImputer
import sklearn.preprocessing, sklearn.decomposition,sklearn.linear_model, sklearn.pipeline, sklearn.metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.impute import SimpleImputer

from sklearn import metrics, svm
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn import utils

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict as cvp
import random
from functools import reduce


# In[2]:


#install sklearn_pandas and scikit-Learn
get_ipython().system('pip install sklearn_pandas')
get_ipython().system('pip install -U scikit-Learn')


# In[3]:


#Read csv file and data cleaning
#df = pd.read_csv("C:\\Users\\sahaj\\Desktop\\AI\\pd_speech_features.csv", delimiter=',',header=None, na_values="?" )
df = pd.read_csv("C:\\Users\\16148\\Desktop\\AI\\pd_speech_features.csv", delimiter=',',header=None, na_values="?" )
df.head()


# In[4]:


#Remove null values
df[df.isnull().any(axis=1)]
#fill null data
df = df.fillna(df.median())


# In[5]:


# Create a LabeL encodes obj ect:
le = LabelEncoder()
le_count = 0 
# ZI:eral:e I:hnough I:he coLurtns
for col in df:
    if df[col].dtype =='object':
        if len(list(df[col].unique())) <= 2:
            le.fit(df[col])
            df[col] = le.transform(df[col]) 
        le_count += 1
print('%d columns were label encoded.' % le_count)


# In[6]:


#change to chategory data
df = df.astype('category')
df.dtypes


# In[7]:


#df.to_csv("C:\\Users\\sahaj\\Desktop\\AI\\pd_speech_features_DF_write.csv")
df.to_csv("C:\\Users\\16148\\Desktop\\AI\\pd_speech_features_DF_write.csv")


# In[8]:


# Read csv file
mnist=genfromtxt("C:\\Users\\16148\\Desktop\\AI\\pd_speech_features_write_category.csv", delimiter=',',skip_header=2)
#mnist=genfromtxt("C:\\Users\\sahaj\\Desktop\\AI\\pd_speech_features_DF_write.csv", delimiter=',',skip_header=2)


# In[9]:


mnist.shape


# In[10]:


n_train = 60000
n_test = 10000

train_idx = np.arange(n_train)
test_idx = np.arange(n_test)+n_train
random.shuffle(train_idx)


# In[11]:


target= mnist[:,1]
data=mnist[:,0]
#print(target)
#print(data)
#print('Data: {}, target: {}'.format(data.shape, target.shape))


# In[12]:


X_train, X_test, y_train, y_test = train_test_split(
    data,
    target,
    test_size=0.2,
    random_state=42,
)


# In[13]:


X_train = X_train[:20000]
y_train = y_train[:20000]
X_test = X_test[:20000]
y_test = y_test[:20000]
X_train = X_train.reshape((len(X_train), 1, 1))
X_test = X_test.reshape((len(X_test), 1, 1))
print('X_train:', X_train.shape, X_train.dtype)
print('y_train:', y_train.shape, y_train.dtype)
print('X_test:', X_test.shape)
print('y_test:', y_test.shape)


# In[14]:


class CascadeForest():
    def __init__(self, base_estimator, params_list, k_fold = 1, evaluate = lambda pre,y: float(sum(pre==y))/len(y)):
        if k_fold > 1: #use cv
            self.params_list = params_list
        else:#use oob
            self.params_list = [params.update({'oob_score':True}) or params for params in params_list]
        self.k_fold = k_fold
        self.evaluate = evaluate
        self.base_estimator = base_estimator
        base_class = base_estimator.__class__
        global prob_class
        class prob_class(base_class): #to use cross_val_predict, estimator's predict method should be predict_prob
            def predict(self, X):
                return base_class.predict_proba(self, X)
        self.base_estimator = prob_class()

    def fit(self,X_train,y_train):
        self.n_classes = len(np.unique(y_train))
        self.estimators_levels = []
        klass = self.base_estimator.__class__
        predictions_levels = []
        self.classes = np.unique(y_train)

        #first level
        estimators = [klass(**params) for params in self.params_list]
        self.estimators_levels.append(estimators)
        predictions = []
        for estimator in estimators:
            estimator.fit(X_train, y_train)
            if self.k_fold > 1:# use cv
                predict_ = cvp(estimator, X_train, y_train, cv=self.k_fold, n_jobs = -1)
            else:
                predict_ = estimator.oob_decision_function_
                #fill default value if meet nan
                inds = np.where(np.isnan(predict_))
                predict_[inds] = 1./self.n_classes
            predictions.append(predict_)
        attr_to_next_level = np.hstack(predictions)
        y_pre = self.classes.take(np.argmax(np.array(predictions).mean(axis=0),axis=1),axis=0)
        self.max_accuracy = self.evaluate(y_pre,y_train)

        #cascade step
        while True:
            print ('level {}, CV accuracy: {}'.format(len(self.estimators_levels),self.max_accuracy))
            estimators = [klass(**params) for params in self.params_list]
            self.estimators_levels.append(estimators)
            predictions = []
            X_train_step = np.hstack((attr_to_next_level,X_train))
            for estimator in estimators:
                estimator.fit(X_train_step, y_train)
                if self.k_fold > 1:
                    predict_ = cvp(estimator, X_train_step, y_train, cv=self.k_fold, n_jobs = -1)
                else:#use oob
                    predict_ = estimator.oob_decision_function_
                    #fill default value if meet nan
                    inds = np.where(np.isnan(predict_))
                    predict_[inds] = 1./self.n_classes
                predictions.append(predict_)
            attr_to_next_level = np.hstack(predictions)
            y_pre = self.classes.take(np.argmax(np.array(predictions).mean(axis=0),axis=1),axis=0)
            accuracy = self.evaluate(y_pre,y_train) 
            final_accuracy= accuracy*100
            print ('Final accuracy {}'.format(final_accuracy))
            if accuracy > self.max_accuracy:
                self.max_accuracy = accuracy
            else:
                self.estimators_levels.pop()
                break

    def predict_proba_staged(self,X):
        #init ouput, shape = nlevel * nsample * nclass
        self.proba_staged = np.zeros((len(self.estimators_levels),len(X),self.n_classes))

        #first level
        estimators = self.estimators_levels[0]
        predictions = []
        for estimator in estimators:
            predict_ = estimator.predict(X)
            predictions.append(predict_)
        attr_to_next_level = np.hstack(predictions)
        self.proba_staged[0] = np.array(predictions).mean(axis=0) 

        #cascade step
        for i in range(1,len(self.estimators_levels)):
            estimators = self.estimators_levels[i]
            predictions = []
            X_step = np.hstack((attr_to_next_level,X))
            for estimator in estimators:
                predict_ = estimator.predict(X_step)
                predictions.append(predict_)
            attr_to_next_level = np.hstack(predictions)
            self.proba_staged[i] = np.array(predictions).mean(axis=0)

        return self.proba_staged
    
    def predict_proba(self,X):
        return self.predict_proba_staged(X)[-1]
    
    def predict_staged(self,X):
        proba_staged = self.predict_proba_staged(X)
        predictions_staged = np.apply_along_axis(lambda proba: self.classes.take(np.argmax(proba),axis=0),
                                                 2, 
                                                 proba_staged)
        return predictions_staged

    def predict(self,X):
        proba = self.predict_proba(X)
        predictions = self.classes.take(np.argmax(proba,axis=1),axis=0) 
        return predictions


# In[15]:


#Scanning Forest
scan_forest_params1 = RandomForestClassifier(n_estimators=30,min_samples_split=21,max_features=1,n_jobs=-1).get_params()
scan_forest_params2 = RandomForestClassifier(n_estimators=30,min_samples_split=21,max_features=1,n_jobs=-1).get_params()

cascade_forest_params1 = RandomForestClassifier(n_estimators=1000,min_samples_split=11,max_features=1,n_jobs=-1).get_params()
cascade_forest_params2 = RandomForestClassifier(n_estimators=1000,min_samples_split=11,max_features=1,n_jobs=-1).get_params()

scan_params_list = [scan_forest_params1,scan_forest_params2]
cascade_params_list = [cascade_forest_params1,cascade_forest_params2]*2


# In[16]:


def calc_accuracy(pre,y):
    return float(sum(pre==y))/len(y)
class ProbRandomForestClassifier(RandomForestClassifier):
    def predict(self, X):
        return RandomForestClassifier.predict_proba(self, X)


# In[17]:


class MultiGrainedScaner():
    def __init__(self, base_estimator, params_list, sliding_ratio = 0.25, k_fold = 3):
        if k_fold > 1: #use cv
            self.params_list = params_list
        else:#use oob
            self.params_list = [params.update({'oob_score':True}) or params for params in params_list]
        self.sliding_ratio = sliding_ratio
        self.k_fold = k_fold
        self.base_estimator = base_estimator
        klass = self.base_estimator.__class__
        self.estimators = [klass(**params) for params in self.params_list]

    #generate scaned samples, X is not None, X[0] is no more than 3d
    def _sample_slicer(self,X,y):
        data_shape = X[0].shape
        window_shape = [max(int(data_size * self.sliding_ratio),1) for data_size in data_shape]
        scan_round_axis = [data_shape[i]-window_shape[i]+1 for i in range(len(data_shape))]
        scan_round_total = reduce(lambda acc,x: acc*x,scan_round_axis)
        if len(data_shape) == 1:
            newX = np.array([x[beg:beg+window_shape[0]]
                                for x in X
                                    for beg in range(scan_round_axis[0])])
        elif len(data_shape) == 2:
            newX = np.array([x[beg0:beg0+window_shape[0],beg1:beg1+window_shape[1]].ravel()
                                for x in X
                                    for beg0 in range(scan_round_axis[0])
                                        for beg1 in range(scan_round_axis[1])])
        elif len(data_shape) == 3:
            newX = np.array([x[beg0:beg0+window_shape[0],beg1:beg1+window_shape[1],beg2:beg2+window_shape[2]].ravel()
                                for x in X
                                    for beg0 in range(scan_round_axis[0])
                                        for beg1 in range(scan_round_axis[1])
                                            for beg2 in range(scan_round_axis[2])])
        newy = y.repeat(scan_round_total)
        return newX,newy,scan_round_total

    #generate new sample vectors
    def scan_fit(self,X,y):
        self.n_classes = len(np.unique(y))
        newX,newy,scan_round_total = self._sample_slicer(X,y)
        sample_vector_list = []
        for estimator in self.estimators:
            estimator.fit(newX, newy)
            if self.k_fold > 1:# use cv
                predict_ = cvp(estimator, newX, newy, cv=self.k_fold, n_jobs = -1)
            else:#use oob
                predict_ = estimator.oob_decision_function_
                #fill default value if meet nan
                inds = np.where(np.isnan(predict_))
                predict_[inds] = 1./self.n_classes
            sample_vector = predict_.reshape((len(X),scan_round_total*self.n_classes))
            sample_vector_list.append(sample_vector)
        return np.hstack(sample_vector_list)

    def scan_predict(self,X):
        newX,newy,scan_round_total = self._sample_slicer(X,np.zeros(len(X)))
        sample_vector_list = []
        for estimator in self.estimators:
            predict_ = estimator.predict(newX)
            sample_vector = predict_.reshape((len(X),scan_round_total*self.n_classes))
            sample_vector_list.append(sample_vector)
        return np.hstack(sample_vector_list)


# In[18]:


train_size = 1000
test_size =1000
shapescanner= 604


# In[19]:


# gcForest 

# Multi-Grained Scan Step
Scaner1 = MultiGrainedScaner(ProbRandomForestClassifier(), scan_params_list, sliding_ratio = 1./4)
Scaner2 = MultiGrainedScaner(ProbRandomForestClassifier(), scan_params_list, sliding_ratio = 1./9)
Scaner3 = MultiGrainedScaner(ProbRandomForestClassifier(), scan_params_list, sliding_ratio = 1./16)

X_train_scan =np.hstack([scaner.scan_fit(X_train[:train_size].reshape((603,1,1)), y_train[:604])
                             for scaner in [Scaner1,Scaner2,Scaner3][:1]])
X_test_scan = np.hstack([scaner.scan_predict(X_test.reshape((len(X_test),1,1)))
                             for scaner in [Scaner1,Scaner2,Scaner3][:1]])

# Cascade RandomForest Step for train data
CascadeRF = CascadeForest(ProbRandomForestClassifier(),cascade_params_list)
CascadeRF.fit(X_train_scan, y_train[:train_size])
X_train_scan =np.hstack([scaner.scan_fit(X_train[:train_size].reshape((603,1,1)), y_train[:604])
                             for scaner in [Scaner1,Scaner2,Scaner3][:1]])
y_pre_staged_train = CascadeRF.predict_staged(X_train_scan)
train_accuracy_staged = np.apply_along_axis(lambda y_pre: calc_accuracy(y_pre,y_train), 1, y_pre_staged_train)
#print('\n'.join('level {}, train accuracy: {}'.format(i+1,train_accuracy_staged[i]) for i in range(len(train_accuracy_staged))))

# Cascade RandomForest Step for test data
X_test_scan = np.hstack([scaner.scan_predict(X_test.reshape((len(X_test),1,1)))
                             for scaner in [Scaner1,Scaner2,Scaner3][:1]])
CascadeRF.fit(X_test_scan, y_test[:test_size])
y_pre_staged_test = CascadeRF.predict_staged(X_test_scan)
test_accuracy_staged = np.apply_along_axis(lambda y_pre: calc_accuracy(y_pre,y_test), 1, y_pre_staged_test)
print('\n'.join('level {}, test accuracy: {}'.format(i+1,test_accuracy_staged[i]) for i in range(len(test_accuracy_staged))))


# In[ ]:




