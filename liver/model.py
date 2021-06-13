import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
df = pd.read_csv("E:/Final year project/liver/liver.csv")
df['Gender']=df['Gender'].apply(lambda x:1 if x=='Male' else 0)
df['Albumin_and_Globulin_Ratio'].mean()
df=df.fillna(0.94)
X=df[['Age', 'Gender', 'Total_Bilirubin', 'Direct_Bilirubin',
       'Alkaline_Phosphotase', 'Alamine_Aminotransferase',
       'Aspartate_Aminotransferase', 'Total_Protiens', 'Albumin',
       'Albumin_and_Globulin_Ratio']]
y=df['Dataset']
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=123)
lr = LogisticRegression()
lr.fit(X_train,y_train)
pickle.dump(lr,open('model.pkl','wb'))
