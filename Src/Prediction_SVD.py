
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sqlite3


# In[2]:


# Create your connection. 
cnx = sqlite3.connect('mortgage.db') 

# Import train and test tables
X_train = pd.read_sql_query("SELECT * FROM X_train", cnx)
X_test = pd.read_sql_query("SELECT * FROM X_test", cnx)
Y_train = pd.read_sql_query("SELECT * FROM Y_train", cnx)
Y_test = pd.read_sql_query("SELECT * FROM Y_test", cnx)


# In[3]:


X_train = X_train.drop('index', axis=1)
X_test = X_test.drop('index', axis=1)
Y_train = Y_train.drop('index', axis=1)
Y_test = Y_test.drop('index', axis=1)

X_train.head(5)


# In[4]:


#Normalization

from sklearn.preprocessing import Normalizer
normalizer = Normalizer()
normalizer.fit(X_train)
train_norm = normalizer.transform(X_train)
test_norm = normalizer.transform(X_test)


# In[5]:


#Standardization

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(train_norm)
train_data = normalizer.transform(train_norm)
test_data = normalizer.transform(test_norm)


# In[6]:


#Dimensionality reduction : Truncated SVD

from sklearn.decomposition import TruncatedSVD
svd = TruncatedSVD(n_components = 68)
svd = svd.fit(train_data)

Train_SVD = svd.transform(train_data)
print(Train_SVD.shape)

Test_SVD = svd.transform(test_data)
print(Test_SVD.shape)


# In[7]:


# Reference: http://fa.bianp.net/blog/2013/different-ways-to-get-memory-consumption-or-lessons-learned-from-memory_profiler/
# return the memory usage in MB 
import psutil
def memory_usage_psutil(): 
    import psutil
    process = psutil.Process(os.getpid()) 
    mem = process.memory_info()[0] / float(2 ** 20)
    return mem


# In[8]:


#decision tree
import time
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

start_time = time.clock()

dec = tree.DecisionTreeClassifier()
dec.fit(Train_SVD,Y_train)
pred_dec = dec.predict(Test_SVD)

Acc_Dec = accuracy_score(Y_test, pred_dec)
print("Decision Tree Accuracy = ", Acc_Dec)

F1_Dec = f1_score(Y_test,pred_dec, average = 'micro')
print("Decision Tree F-1 score(micro) = ", F1_Dec)

F1W_Dec = f1_score(Y_test,pred_dec, average = 'weighted')
print("Decision Tree F-1 score(weighted) = ",F1W_Dec )

Time_Dec = time.clock() - start_time
print("Time Taken in seconds:", Time_Dec)

Mem_Dec = memory_usage_psutil()
print("Decision Tree Memory usage: ",Mem_Dec)

Cpu_Dec = psutil.cpu_percent()
print("Decision Tree Cpu Percent: ", Cpu_Dec)


# In[9]:


#Classification algorithm KNN

from sklearn.neighbors import KNeighborsClassifier

start_time = time.clock()
knn = KNeighborsClassifier(n_neighbors=1000, weights='uniform', algorithm='auto')
knn.fit(Train_SVD,Y_train)
pred_KNN = knn.predict(Test_SVD)

Acc_Knn = accuracy_score(Y_test, pred_KNN)
print("KNN Accuracy = ",Acc_Knn)

F1_Knn = f1_score(Y_test,pred_KNN,average = 'micro')
print("KNN F-1 score(micro) = ", F1_Dec)

F1W_Knn = f1_score(Y_test,pred_KNN,average = 'weighted')
print("KNN F-1 score(weighted) = ", F1W_Dec)

Time_Knn = time.clock() - start_time
print("Time Taken in seconds:", Time_Knn) 

Mem_Knn = memory_usage_psutil()
print("KNN Memory usage: ",Mem_Knn)

Cpu_Knn = psutil.cpu_percent()
print("KNN Cpu Percent: ", Cpu_Knn)


# In[10]:


#SGD Classifier

from sklearn.linear_model import SGDClassifier

start_time = time.clock()
SGD_model = SGDClassifier().fit(Train_SVD,Y_train)
pred_sgd = SGD_model.predict(Test_SVD)

Acc_Sgd = accuracy_score(Y_test, pred_sgd)
print("SGD Accuracy=", Acc_Sgd)

F1_Sgd = f1_score(Y_test, pred_sgd, average = 'micro')
print("SGD F-1 score(micro) = ", F1_Sgd)

F1W_Sgd = f1_score(Y_test, pred_sgd, average = 'weighted')
print("SGD F-1 score(weighted) = ", F1W_Sgd)

Time_Sgd = time.clock() - start_time
print("Time Taken in seconds:", Time_Sgd) 

Mem_Sgd = memory_usage_psutil()
print("SGD Memory usage: ",Mem_Sgd)

Cpu_Sgd = psutil.cpu_percent()
print("SGD Cpu Percent: ", Cpu_Sgd)


# In[11]:


#Naive Bayes Classifier

from sklearn.naive_bayes import GaussianNB

start_time = time.clock()
GaussianNB_model = GaussianNB().fit(Train_SVD,Y_train)
pred_naive = GaussianNB_model.predict(Test_SVD)

Acc_Naive = accuracy_score(Y_test, pred_naive)
print("Naive bayes Accuracy = ", Acc_Naive)

F1_Naive = f1_score(Y_test, pred_naive, average= 'micro')
print("Naive bayes F-1 score(micro) = ", F1_Naive)

F1W_Naive = f1_score(Y_test, pred_naive, average= 'weighted')
print("Naive bayes F-1 score(weighted) = ", F1W_Naive)

Time_Naive = time.clock() - start_time
print("Time Taken in seconds:", Time_Naive) 

Mem_Naive = memory_usage_psutil()
print("Naive bayes Memory usage: ",Mem_Naive)

Cpu_Naive = psutil.cpu_percent()
print("Naive bayes Cpu Percent: ", Cpu_Naive)


# In[12]:


#MLP CLassifier

from sklearn.neural_network import MLPClassifier

start_time = time.clock()
MLP_model = MLPClassifier(solver='sgd', hidden_layer_sizes=(5,), random_state=1)
MLP_model = MLP_model.fit(Train_SVD,Y_train)
pred_mlp = MLP_model.predict(Test_SVD)

Acc_Mlp = accuracy_score(Y_test, pred_mlp)
print("MLP Accuracy=", Acc_Mlp)

F1_Mlp = f1_score(Y_test, pred_mlp, average= 'micro')
print("MLP F-1 score(micro) = ", F1_Mlp)

F1W_Mlp = f1_score(Y_test, pred_mlp, average= 'weighted')
print("MLP F-1 score(weighted) = ", F1W_Mlp)

Time_Mlp = time.clock() - start_time
print("Time Taken in seconds:", Time_Mlp)

Mem_Mlp = memory_usage_psutil()
print("MLP Memory usage: ",Mem_Mlp)

Cpu_Mlp = psutil.cpu_percent()
print("MLP Cpu Percent: ", Cpu_Mlp)


# In[13]:


#GradientBoost

from sklearn.ensemble import GradientBoostingClassifier

start_time = time.clock()
grad = GradientBoostingClassifier(n_estimators=100)
grad = grad.fit(Train_SVD,Y_train)
pred_grad = grad.predict(Test_SVD)

Acc_Grad = accuracy_score(Y_test, pred_grad)
print("Gradient Boost accuracy = ", Acc_Grad)

F1_Grad = f1_score(Y_test,pred_grad, average= 'micro')
print("Gradient Boost F-1 score(micro) = ", F1_Grad)

F1W_Grad = f1_score(Y_test,pred_grad, average= 'weighted')
print("Gradient Boost F-1 score(weighted) = ", F1W_Grad)

Time_Grad = time.clock() - start_time
print("Time Taken in seconds:", Time_Grad) 

Mem_Grad = memory_usage_psutil()
print("Gradient Boost Memory usage: ",Mem_Grad)

Cpu_Grad = psutil.cpu_percent()
print("Gradient Boost Cpu Percent: ", Cpu_Grad)


# In[14]:


#Random Forest Classifier

from sklearn.ensemble import RandomForestClassifier

start_time = time.clock()
RF = RandomForestClassifier(n_estimators=100,n_jobs=-1,criterion='gini',class_weight = 'balanced')
RF = RF.fit(Train_SVD,Y_train)
pred_RF = RF.predict(Test_SVD)

Acc_RF = accuracy_score(Y_test, pred_RF)
print("Random forest Accuracy = ", Acc_RF)

F1_RF = f1_score(Y_test,pred_RF,average= 'micro')
print("Random forest F-1 score(micro) = ", F1_RF)

F1W_RF = f1_score(Y_test,pred_RF,average= 'weighted')
print("Random forest F-1 score(weighted) = ", F1W_RF)

Time_RF = time.clock() - start_time
print("Time Taken in seconds:",Time_RF) 

Mem_RF = memory_usage_psutil()
print("Random forest Memory usage: ",Mem_RF)

Cpu_RF = psutil.cpu_percent()
print("Random forest Cpu Percent: ", Cpu_RF)


# In[15]:


#Extra Tress

from sklearn.ensemble import ExtraTreesClassifier

start_time = time.clock()
extra_trees = ExtraTreesClassifier(n_estimators=100,n_jobs=-1,criterion='gini',class_weight = 'balanced')
extra_trees = extra_trees.fit(Train_SVD,Y_train)
pred_ext = extra_trees.predict(Test_SVD)

Acc_Ext = accuracy_score(Y_test, pred_ext)
print("Extra Trees Accuracy=", Acc_Ext)

F1_Ext = f1_score(Y_test, pred_ext, average= 'micro')
print("Extra Trees F-1 score(micro) = ", F1_Ext)

F1W_Ext = f1_score(Y_test, pred_ext, average= 'weighted')
print("Extra Trees F-1 score(weighted) = ", F1W_Ext)

Time_Ext = time.clock() - start_time
print("Time Taken in seconds:", Time_Ext)

Mem_Ext = memory_usage_psutil()
print("Extra Trees Memory usage: ",Mem_Ext)

Cpu_Ext = psutil.cpu_percent()
print("Extra Trees Cpu Percent: ", Cpu_Ext)


# In[16]:


#Adaboost

from sklearn.ensemble import AdaBoostClassifier,RandomForestClassifier

start_time = time.clock()
AdaB_model= AdaBoostClassifier(RandomForestClassifier(n_estimators=100,n_jobs=-1,criterion='gini',class_weight = 'balanced'))
AdaB_model = AdaB_model.fit(Train_SVD,Y_train)
pred_Adab = AdaB_model.predict(Test_SVD)

Acc_Adab = accuracy_score(Y_test, pred_Adab)
print("Adaboost accuracy =", Acc_Adab)

F1_Adab = f1_score(Y_test,pred_Adab,average= 'micro')
print("Adaboost F-1 score(micro) = ", F1_Adab)

F1W_Adab = f1_score(Y_test,pred_Adab,average= 'weighted')
print("Adaboost F-1 score(weighted) = ", F1W_Adab )

Time_Adab = time.clock() - start_time
print("Time Taken in seconds:", Time_Adab)

Mem_Adab = memory_usage_psutil()
print("Adaboost Memory usage: ",Mem_Adab)

Cpu_Adab = psutil.cpu_percent()
print("Adaboost Cpu Percent: ", Cpu_Adab)


# In[17]:


#XGBoost

from xgboost import XGBClassifier

start_time = time.clock()
XGB_model= XGBClassifier()
XGB_model = XGB_model.fit(Train_SVD,Y_train)
pred_XGB = XGB_model.predict(Test_SVD)

Acc_XGB = accuracy_score(Y_test, pred_XGB)
print("XGB accuracy =", Acc_XGB)

F1_XGB = f1_score(Y_test,pred_XGB,average= 'micro')
print("XGB F-1 score(micro) = ", F1_XGB)

F1W_XGB = f1_score(Y_test,pred_XGB,average= 'weighted')
print("XGB F-1 score(weighted) = ", F1W_XGB )

Time_XGB = time.clock() - start_time
print("Time Taken in seconds:", Time_XGB) 

Mem_XGB = memory_usage_psutil()
print("XGB Memory usage: ",Mem_XGB)

Cpu_XGB = psutil.cpu_percent()
print("XGB Cpu Percent: ", Cpu_XGB)


# In[18]:


#BarChart to compare different Accuracies

Acc =[]
label = []

Acc = [Acc_Dec, Acc_Knn, Acc_Sgd, Acc_Naive, Acc_Mlp, Acc_Grad, Acc_RF, Acc_Ext, Acc_Adab, Acc_XGB]
label = ['Decision','KNN','SGD','NB','MLP','GradDesc','RandFor','ExtTree','Adaboost','XGBoost']

x_axis = np.arange(len(Acc))
y_axis = Acc

plt.bar(x_axis, y_axis, color = 'green')
plt.xticks(x_axis, label, fontsize=10, rotation=30)

plt.xlabel("Algorithms")
plt.ylabel('Accuracy')
plt.title('Accuracy Comparison Between All Algorithms (SVD)')

plt.savefig('/Users/vineevineela/Downloads/220_Assignment/Images/Accuracy_SVD.svg',bbox_inches = "tight")
plt.clf()


# In[19]:


#BarChart to compare different F-1 Scores(Micro)

F1_score =[]
label = []

F1_score = [F1_Dec, F1_Knn, F1_Sgd, F1_Naive, F1_Mlp, F1_Grad, F1_RF, F1_Ext, F1_Adab, F1_XGB]
label = ['Decision','KNN','SGD','NB','MLP','GradDesc','RandFor','ExtTree','Adaboost','XGBoost']

x_axis = np.arange(len(F1_score))
y_axis = F1_score

plt.bar(x_axis, y_axis, color = 'orange')
plt.xticks(x_axis, label, fontsize=10, rotation=30)

plt.xlabel("Algorithms")
plt.ylabel('F1_score(Micro)')
plt.title('F1_score(Micro) Comparison Between All Algorithms  (SVD)')

plt.savefig('/Users/vineevineela/Downloads/220_Assignment/Images/F1_Micro_SVD.svg',bbox_inches = "tight")
plt.clf()


# In[20]:


#BarChart to compare different F-1 Scores(Weighted)

F1W_score =[]
label = []

F1W_score = [F1W_Dec, F1W_Knn, F1W_Sgd, F1W_Naive, F1W_Mlp, F1W_Grad, F1W_RF, F1W_Ext, F1W_Adab, F1W_XGB]
label = ['Decision','KNN','SGD','NB','MLP','GradDesc','RandFor','ExtTree','Adaboost','XGBoost']

x_axis = np.arange(len(F1W_score))
y_axis = F1W_score

plt.bar(x_axis, y_axis, color = 'purple')
plt.xticks(x_axis, label, fontsize=10, rotation=30)

plt.ylabel('F1W_score(Weighted)')
plt.xlabel("Algorithms")

plt.title('F1_score(Weighted) Comparison Between All Algorithms  (SVD)')

plt.savefig('/Users/vineevineela/Downloads/220_Assignment/Images/F1_Weighted_SVD.svg',bbox_inches = "tight")
plt.clf()


# In[21]:


#Line Graph
#Reference: https://python-graph-gallery.com/122-multiple-lines-chart/

plt.plot( x_axis, Acc, color='purple', linewidth = 8, linestyle='dotted', label="Accuracy")
plt.plot( x_axis, F1_score, color='orange', linewidth=3,linestyle='dashed', label="F1_score")
plt.plot( x_axis, F1W_score, color='green', linewidth=2, linestyle='dashed', label="F1W_score")
plt.xticks(x_axis, label, fontsize=10, rotation=30)

plt.xlabel("Algorithms")
plt.title('Comparison of All Metrics(SVD)')
plt.legend()

plt.savefig('/Users/vineevineela/Downloads/220_Assignment/Images/Metrics_SVD.svg',bbox_inches = "tight")
plt.clf()


# In[22]:


#Line Graph
#Reference: https://python-graph-gallery.com/122-multiple-lines-chart/

Cpu_Per = []

Cpu_Per = [Cpu_Dec, Cpu_Knn, Cpu_Sgd, Cpu_Naive, Cpu_Mlp, Cpu_Grad, Cpu_RF, Cpu_Ext, Cpu_Adab, Cpu_XGB]
plt.plot( x_axis, Cpu_Per, color='cyan', linewidth=3, linestyle='dashed', label="Cpu_Per")

plt.xticks(x_axis, label, fontsize=10, rotation=30)
plt.xlabel("Algorithms")
plt.ylabel("CPU Utilization Percentage")
plt.title('CPU Per(%) Comparison between all algorithms (PCA)')

plt.savefig('/Users/vineevineela/Downloads/220_Assignment/Images/CPU_Per_SVD.svg',bbox_inches = "tight")
plt.clf()


# In[23]:


Mem_usage = []

Mem_usage = [Mem_Dec, Mem_Knn, Mem_Sgd, Mem_Naive, Mem_Mlp, Mem_Grad, Mem_RF, Mem_Ext, Mem_Adab, Mem_XGB]
plt.plot( x_axis, Mem_usage, color='magenta', linewidth=3,linestyle='dashed', label="Mem_usage")

plt.xticks(x_axis, label, fontsize=10, rotation=30)
plt.xlabel("Algorithms")
plt.ylabel("Memory usage")
plt.title('Memory Usage Comparison between all algorithms (PCA)')

plt.savefig('/Users/vineevineela/Downloads/220_Assignment/Images/Mem_Usage_SVD.svg',bbox_inches = "tight")
plt.clf()


# In[24]:


# Time comparison
Time = []
Time = [Time_Dec, Time_Knn, Time_Sgd, Time_Naive, Time_Mlp, Time_Grad, Time_RF, Time_Ext, Time_Adab, Time_XGB] 

plt.bar(label,Time, color = 'red')
plt.xticks(x_axis, label, fontsize=10, rotation=30)
plt.ylabel('Latency Time')
plt.xlabel("Algorithms")
plt.title('Latency Time Comparison Between All Algorithms (PCA)')

plt.savefig('/Users/vineevineela/Downloads/220_Assignment/Images/Latency_Time_SVD.svg',bbox_inches = "tight")
plt.clf()


# In[25]:


#Individual Line graphs
#plt.plot(x_axis, Acc, color='red')
#plt.xlabel('Accuracies')
#plt.xticks(x_axis, label, fontsize=10, rotation=30)
#plt.title('Accuracy Comparison')
#plt.show()


# In[26]:


#Cross_validation for the best algorithm (Decision Tree) with 3 folds

from sklearn.model_selection import cross_val_score

Dec_cv = cross_val_score(dec,Train_SVD,Y_train,cv=3).mean()

print('Decision Tree cross validation score', Dec_cv)


# In[27]:


#Report For Decision Tree
from sklearn import metrics
from sklearn.metrics import classification_report

print(classification_report(Y_test,pred_dec))


# In[28]:


#Confusion Matrix and Report For Decision Tree
from sklearn.metrics import confusion_matrix

print('Confusion Matrix')
print(confusion_matrix(Y_test, pred_dec))


# In[29]:


# Top features affecting classification
# make importance relative to the max importance

fimportances = dec.feature_importances_
feature_importance = 100.0 * (fimportances / fimportances.max())
sorted_idex = np.argsort(feature_importance)
feature_names = list(X_test.columns.values)
feature_sorted = [feature_names[indice] for indice in sorted_idex]
pos = np.arange(sorted_idex.shape[0]) + .5
print 'Top 10 features are: '
for feature in feature_sorted[::-1][:10]:
    print feature

