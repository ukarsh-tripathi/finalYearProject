import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
df = pd.read_csv("kidney.csv")
x = df.drop(['Class'],axis=1)
y = df['Class']
lab_enc=LabelEncoder()
y=lab_enc.fit_transform(y)
from sklearn.ensemble import RandomForestClassifier
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.3, random_state=1)
clf=RandomForestClassifier(n_estimators=100)
clf.fit(X_train,Y_train)
clf=RandomForestClassifier(n_estimators=100)
clf.fit(X_train,Y_train)
y_predi=clf.predict(X_test)
#from sklearn.metrics import accuracy_score
#print("Accuracy on training dataset:{:.1f}".format(clf.score(X_train,Y_train)))
#print("Accuracy on testing dataset:{:.1f}".format(accuracy_score(Y_test, y_predi)))
pickle.dump(clf,open('model.pkl','wb'))