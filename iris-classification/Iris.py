import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

df=pd.read_csv('11-iris.csv')
print(df.head())
print(df.info())
print(df.isnull().sum())

df=df.drop("Id",axis=1)

sns.pairplot(df,hue="Species")
#plt.show()

from sklearn.preprocessing import LabelEncoder
label_encoder=LabelEncoder()
df["Species"]=label_encoder.fit_transform(df.Species)
print(df["Species"].value_counts())

X=df.drop("Species",axis=1)
y=df["Species"]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42,stratify=y)#stratify dagılım oranını korur mesela %33 %33 %33 olarak test verisine dagıtır !!

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)

from sklearn.linear_model import LogisticRegression
log_reg=LogisticRegression()
log_reg.fit(X_train,y_train)
y_pred=log_reg.predict(X_test)

from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
print("--- Logistic Regression --- ")
print("Accuracy Score",accuracy_score(y_test,y_pred))
print("Classification Report",classification_report(y_test,y_pred))
print("Confusion Matrix",confusion_matrix(y_test,y_pred))

from sklearn.tree import DecisionTreeClassifier
tree=DecisionTreeClassifier()
tree.fit(X_train,y_train)
y_pred=tree.predict(X_test)
print("--- Decision Tree Classifier --- ")
print("Accuracy Score",accuracy_score(y_test,y_pred))
print("Classification Report",classification_report(y_test,y_pred))
print("Confusion Matrix",confusion_matrix(y_test,y_pred))

from sklearn.ensemble import RandomForestClassifier
forest=RandomForestClassifier()
forest.fit(X_train,y_train)
y_pred=forest.predict(X_test)
print("--- Random Forest Classifier --- ")
print("Accuracy Score",accuracy_score(y_test,y_pred))
print("Classification Report",classification_report(y_test,y_pred))
print("Confusion Matrix",confusion_matrix(y_test,y_pred))

from sklearn.ensemble import GradientBoostingClassifier
gbt=GradientBoostingClassifier()
gbt.fit(X_train,y_train)
y_pred=gbt.predict(X_test)
print("--- Gradient Boosting Classifier --- ")
print("Accuracy Score",accuracy_score(y_test,y_pred))
print("Classification Report",classification_report(y_test,y_pred))
print("Confusion Matrix",confusion_matrix(y_test,y_pred))

from sklearn.ensemble import AdaBoostClassifier
adaboost=AdaBoostClassifier()
adaboost.fit(X_train,y_train)
y_pred=adaboost.predict(X_test)
print("--- AdaBoost Classifier --- ")
print("Accuracy Score",accuracy_score(y_test,y_pred))
print("Classification Report",classification_report(y_test,y_pred))
print("Confusion Matrix",confusion_matrix(y_test,y_pred))


#hyperparameter tuning
from sklearn.model_selection import GridSearchCV
grid=GridSearchCV(
    estimator=LogisticRegression(),
    param_grid={
        "penalty": ["l1", "l2"],
        "solver":["liblinear","lbfgs","newton-cg"],
    },
    n_jobs=1,
    cv=5
)
grid.fit(X_train,y_train)
y_pred=grid.predict(X_test)
print("--- Logistic Regression(After) --- ")
print("Accuracy Score",accuracy_score(y_test,y_pred))
print("Classification Report",classification_report(y_test,y_pred))
print("Confusion Matrix",confusion_matrix(y_test,y_pred))
