import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

#load dataset
df = pd.read_csv('synthetic_diabetes_dataset_300.csv')

X = df.drop('Outcome', axis='columns')
y = df['Outcome']

#split data into train and test parts
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=11)

#train SVC
svc_clf = SVC()
svc_clf.fit(X_train, y_train)

#make predictions
y_pred = svc_clf.predict(X_test)

#SVC model evaluation
accuracy =  accuracy_score(y_test, y_pred)
clf_report = classification_report(y_test, y_pred)

#printing dataset
print(df)

#printing metrics
print('Accuracy:', accuracy)
print('Classification report:', clf_report)