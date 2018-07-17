import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

df = pd.read_csv('KNN_Project_Data',index_col=0)

scaler = StandardScaler()
scaler.fit(df.drop('TARGET CLASS',axis=1))
scaled_features = scaler.transform(df.drop('TARGET CLASS',axis=1))

df_features = pd.DataFrame(scaled_features,columns=df.columns[:-1])

X = scaled_features
y = df['TARGET CLASS']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=101)

KNN = KNeighborsClassifier(n_neighbors=31)
KNN.fit(X_train,y_train)

predictions = KNN.predict(X_test)

print(classification_report(y_test,predictions))

#Using the elbow method to choose the best value of K

error_rate = []

for i in range(1,40):
	KNN = KNeighborsClassifier(n_neighbors=i)
	KNN.fit(X_train,y_train)
	pred_i = KNN.predict(X_test)
	error_rate.append(np.mean(pred_i != y_test))

plt.plot(range(1,40),error_rate)
plt.xlabel('K Values')
plt.ylabel('Error_Rate')
plt.show()