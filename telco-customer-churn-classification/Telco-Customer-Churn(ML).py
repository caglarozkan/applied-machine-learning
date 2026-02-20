import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
#print(df.head())
#print(df.columns)
#print(df.describe)

print(df.info())
for col in df.columns:
    print(f"\nColumn {col}: {df[col].value_counts()}")

df.drop("customerID", axis=1, inplace=True) #id is not important bcs each row is unique

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
print("\nNull num in Total Charges after transition:",df["TotalCharges"].isnull().sum())
df["TotalCharges"]=df["TotalCharges"].fillna(df["TotalCharges"].median())
print("\nNull num in Total Charges after filling:",df["TotalCharges"].isnull().sum())

df["Churn"]=df["Churn"].map({"Yes":1,"No":0})

print(df.info())

binary_columns=[
    "gender","Partner","Dependents",
    "PhoneService","PaperlessBilling"
]
binary_map={
    "Yes":1,
    "No":0,
    "Male":1,
    "Female":0
}
for col in binary_columns:
    df[col]= df[col].map(binary_map)

service_cols = [
    "MultipleLines", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies"
]
df = pd.get_dummies(df, columns=service_cols, drop_first=True)

df = pd.get_dummies(
    df,
    columns=["InternetService", "Contract", "PaymentMethod"],
    drop_first=True
)
#print(df.info())

#for powerful but basic baseline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
X=df.drop(["Churn"], axis=1)
y=df["Churn"]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=20,stratify=y)#inbalanced data, train ve testte dengeli bi sonuca ulaşabilmek için kullanyoruz

log_reg=LogisticRegression()
log_reg.fit(X_train,y_train)
y_pred=log_reg.predict(X_test)

from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
print("Accuracy:",accuracy_score(y_pred,y_test))
print("Confusion Matrix:",confusion_matrix(y_pred,y_test))
print("Classification Report:",classification_report(y_pred,y_test))

model_balanced=LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
)
model_balanced.fit(X_train,y_train)
y_pred=model_balanced.predict(X_test)

print("Accuracy:",accuracy_score(y_pred,y_test))
print("Confusion Matrix:",confusion_matrix(y_pred,y_test))
print("Classification Report:",classification_report(y_pred,y_test))

y_proba = model_balanced.predict_proba(X_test)[:,1]
y_pred_custom = (y_proba >= 0.35).astype(int)
print("Confusion Matrix:",confusion_matrix(y_pred_custom,y_test))
print("Classification Report:",classification_report(y_pred_custom,y_test))

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    n_estimators=200,
    random_state=20,
    class_weight="balanced"
)

rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))
print("Classification Report:\n", classification_report(y_test, y_pred_rf))

y_proba_rf = rf.predict_proba(X_test)[:,1]
y_pred_rf_035 = (y_proba_rf >= 0.35).astype(int)

print(confusion_matrix(y_test, y_pred_rf_035))
print(classification_report(y_test, y_pred_rf_035))

importances = pd.Series(
    rf.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

print(importances.head(10))

importances.head(10).plot(kind="barh")
plt.title("Top 10 Feature Importance")
plt.show()


from sklearn.model_selection import GridSearchCV
param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [None, 5, 10],
    "min_samples_split": [2, 5, 10]
}

grid = GridSearchCV(
    RandomForestClassifier(
        random_state=20,
        class_weight="balanced"
    ),
    param_grid,
    scoring="recall",
    cv=5,
    n_jobs=-1
)
#neden gridSearch:Baseline Random Forest modelinin churn recall’ını artırmak için hyperparameter tuning yaptım ve scoring olarak recall kullandım çünkü churn kaçırmak iş açısından daha maliyetli
grid.fit(X_train, y_train)

print("grid best params:",grid.best_params_)
print("Best recall:", grid.best_score_)
best_rf = grid.best_estimator_

y_pred_best = best_rf.predict(X_test)

print(confusion_matrix(y_test, y_pred_best))
print(classification_report(y_test, y_pred_best))
best_rf = grid.best_estimator_

print("Best params:", grid.best_params_)
print("Best CV recall:", grid.best_score_)


y_proba_best = best_rf.predict_proba(X_test)[:,1]

for t in [0.5, 0.4, 0.35, 0.3]:
    print(f"\n--- Threshold: {t} ---")
    y_pred_t = (y_proba_best >= t).astype(int)
    print(confusion_matrix(y_test, y_pred_t))
    print(classification_report(y_test, y_pred_t))


from sklearn.metrics import roc_curve, auc

fpr, tpr, _ = roc_curve(y_test, y_proba_best)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0,1],[0,1],'--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Final Random Forest")
plt.legend()
plt.show()


final_importances = pd.Series(
    best_rf.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

final_importances.head(10).plot(kind="barh")
plt.title("Top 10 Feature Importance - Final Model")
plt.show()