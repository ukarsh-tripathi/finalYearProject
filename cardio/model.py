import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
df = pd.read_csv("cardio.csv", sep=";")
df['age'] = df['age'].map(lambda x : x // 365)
from sklearn.model_selection import train_test_split
X = df.drop(['cardio'], axis=1)
y = df['cardio']
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42) 
lr = LogisticRegression()
lr.fit(X_train,y_train)
pickle.dump(lr,open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))
print(model.predict([[0,45,2,168,62.0,110,80,1,1,0,0,1]]))