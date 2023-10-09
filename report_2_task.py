from gettext import install
import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta,date 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.svm import SVC
import plotly.express as px
import plotly.graph_objects as go


data = pd.read_csv('covid19.csv')
df= data.fillna(0)

df1 = df.drop(['percentoday','ratedeaths','numdeathstoday','percentdeath','numtestedtoday','numteststoday','numrecoveredtoday', 'percentactive','numactive','rateactive','numtotal_last14','ratetotal_last14','numdeaths_last14','ratedeaths_last14','numtotal_last7', 'ratetotal_last7','numdeaths_last7','ratedeaths_last7','avgtotal_last7','avgincidence_last7','avgdeaths_last7','avgratedeaths_last7','raterecovered'],1)
cor = df1.corr()


fig, ax = plt.subplots(figsize=(20, 20))
yticks = df1.index
keptticks = yticks[::int(len(yticks)/10)]
yticks = ['' for y in yticks]
yticks[::int(len(yticks)/10)] = keptticks

xticks = df1.columns
keptticks = xticks[::int(len(xticks)/10)]
xticks = ['' for y in xticks]
xticks[::int(len(xticks)/10)] = keptticks
sns.heatmap(cor, annot=True)
plt.yticks(rotation=0) 


X = df.drop(['prname','prnameFR','date'],1)
y = df['numdeaths']




X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=42)

#KNN 
knn = KNeighborsClassifier(n_neighbors=2)  
knn.fit(X_train, y_train)

#Decisiontree
from sklearn.tree import DecisionTreeClassifier
clf_entropy = DecisionTreeClassifier(criterion = "entropy", random_state = 200,max_depth = 3, min_samples_leaf = 5)
clf_entropy.fit(X_train, y_train)
my_predictions  = clf_entropy.predict(X_test)

from sklearn.metrics import accuracy_score, confusion_matrix

#Random forest
from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier(n_estimators=50, random_state=44)
rf_model.fit(X_train, y_train)
y_pred_test = rf_model.predict(X_test)


# Predict on dataset which model has not seen before
accuracy_score(y_test, y_pred_test)
accuracy = round(clf_entropy.score(X_train, y_train) * 100, 2)
print("Decision tree Model Accuracy:- ", accuracy)
print("KNN Model Accuracy ",knn.score(X_test, y_test)*100)
print("Randomforest Model Accuracy ",rf_model.score(X_test, y_test)*100)


fig1 = go.Figure([go.Scatter(x=df1['date'], y=y_test)])
fig1.update_layout(title_text='Predicted number of death cases for each date using KNN')

fig2 = go.Figure([go.Scatter(x=df1['date'], y=y_pred_test)])
fig2.update_layout(title_text='Predicted number of death cases for each date using Random forest')


fig1.show()
fig2.show()
plt.show()