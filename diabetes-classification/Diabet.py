import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv('diabetes.csv')
"""print(df.head())
print(df.info())
print(df.isnull().sum())

print(df["Outcome"].value_counts())

sns.scatterplot(data=df,x="Age",y="Insulin",hue="Outcome")
plt.show()
"""
import math
def plot_all_histograms(df, title_prefix=""):
    num_cols = df.select_dtypes(include=[np.number]).columns
    n_cols = 3
    n_rows = math.ceil(len(num_cols) / n_cols)

    plt.figure(figsize=(5 * n_cols, 4 * n_rows))

    for i, col in enumerate(num_cols, 1):
        plt.subplot(n_rows, n_cols, i)
        sns.histplot(df[col], kde=True, bins=30)
        plt.title(f"{title_prefix}{col}")
        plt.xlabel("")
        plt.ylabel("")

    plt.tight_layout()
    plt.show()

print(plot_all_histograms(df))

#Insulin,BloodPressure,SkinThickness,Glucose,BMI   0 ---> has zeros
zeros_col=["Insulin","BloodPressure","SkinThickness","Glucose","BMI"]
for col in zeros_col:
    zero_count = (df[col] == 0).sum()
    total_count = df[col].count()

    percentage = zero_count / total_count * 100

    print(f"{col} :0 olan değerlerin yüzdesi: %{percentage:.2f}")
df=df.drop("Insulin",axis=1)

X=df.drop("Outcome",axis=1)
y=df["Outcome"]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42,stratify=y)

#filling 0 values with median
columns_to_fill=["Glucose","BloodPressure","SkinThickness","BMI"]
medians={}
for column in columns_to_fill:
    median_value=X_train[X_train[column]!=0][column].median()
    medians[column]=median_value
    X_train[column]=X_train[column].replace(0,median_value)

for column in columns_to_fill:
    X_test[column]=X_test[column].replace(0,medians[column])

#Scaling
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)

from sklearn.metrics import roc_auc_score

from sklearn.linear_model import LogisticRegression
log_reg=LogisticRegression()
log_reg.fit(X_train,y_train)
y_pred=log_reg.predict(X_test)

from sklearn.metrics import classification_report,accuracy_score,confusion_matrix
print("--- LOGISCTIC REG ---")
print("Accuracy Score",accuracy_score(y_test,y_pred))
print("Classification Report:",classification_report(y_test,y_pred))
print("Confusion Matrix",confusion_matrix(y_test,y_pred))
y_proba = log_reg.predict_proba(X_test)[:,1]
print("Logistic ROC-AUC:", roc_auc_score(y_test, y_proba))

from sklearn.tree import DecisionTreeClassifier
dec_tree=DecisionTreeClassifier()
dec_tree.fit(X_train,y_train)
y_pred=dec_tree.predict(X_test)

print("--- Decision Tree ---")
print("Accuracy Score",accuracy_score(y_test,y_pred))
print("Classification Report:",classification_report(y_test,y_pred))
print("Confusion Matrix",confusion_matrix(y_test,y_pred))
y_proba = dec_tree.predict_proba(X_test)[:,1]
print("Decision Tree ROC-AUC:", roc_auc_score(y_test, y_proba))

from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier()
knn.fit(X_train,y_train)
y_pred=knn.predict(X_test)
print("--- KNN ---")
print("Accuracy Score",accuracy_score(y_test,y_pred))
print("Classification Report",classification_report(y_test,y_pred))
print("Confusion Matrix",confusion_matrix(y_test,y_pred))
y_proba = knn.predict_proba(X_test)[:,1]
print("KNN ROC-AUC:", roc_auc_score(y_test, y_proba))

from xgboost import XGBClassifier
xgbc=XGBClassifier()
xgbc.fit(X_train,y_train)
y_pred=xgbc.predict(X_test)

print("--- XGBoost ---")
print("Accuracy Score",accuracy_score(y_test,y_pred))
print("Classification Report",classification_report(y_test,y_pred))
print("Confusion Matrix",confusion_matrix(y_test,y_pred))
y_proba = xgbc.predict_proba(X_test)[:,1]
print("XGB ROC-AUC:", roc_auc_score(y_test, y_proba))

from sklearn.model_selection import GridSearchCV

param_grid = {
    "max_depth": [2,3,4,5],
    "learning_rate": [0.01, 0.05, 0.1],
    "n_estimators": [100,200,300],
    "subsample": [0.8,1],
}

grid = GridSearchCV(
    XGBClassifier(random_state=42, eval_metric="logloss"),
    param_grid,
    scoring="recall",   # 🔥 burası önemli
    cv=5,
    n_jobs=-1
)

grid.fit(X_train, y_train)

print("--- XGBOOST (after tuning) ---")
print("Accuracy Score",accuracy_score(y_test,y_pred))
print("Classification Report",classification_report(y_test,y_pred))
print("Confusion Matrix",confusion_matrix(y_test,y_pred))
y_proba = grid.predict_proba(X_test)[:,1]
print("XGB ROC-AUC:", roc_auc_score(y_test, y_proba))

"""
Recall=TP/TP+FN
"""