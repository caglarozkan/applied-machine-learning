import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")

df=pd.read_csv("insurance.csv")

print(df.head())
print(df.info())
print(df.isnull().sum())

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

df["sex"]=df["sex"].map({"male":0, "female":1})
df["smoker"]=df["smoker"].map({"yes":0, "no":1})

print(df["region"].value_counts())

"""from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
df["region"] = labelencoder.fit_transform(df["region"])
print(df["region"].value_counts())"""
#bu şekilde yaptıgımızda mesela southwest 1 northwest 2 olursa sanki northwest daha iyiymişş gibi olur o yüzden get dummies yapmak daha iyi
df=pd.get_dummies(df, columns=["region"], drop_first=True)


X=df.drop("charges", axis=1)
y=df["charges"]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#scaling gerektirmez tree modelleri
"""from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)"""

from sklearn.tree import DecisionTreeRegressor
dec_tree_reg = DecisionTreeRegressor(random_state=42)
dec_tree_reg.fit(X_train, y_train)
y_pred = dec_tree_reg.predict(X_test)

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

print("\n Decision Tree Regressor --- ")
print("R2 Score:",r2_score(y_test, y_pred))
print("Mean Absolute Error:",mean_absolute_error(y_test, y_pred))
print("Mean Squared Error:",mean_squared_error(y_test, y_pred))

from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
print("\nRandom Forest Regressor --- ")
print("R2 Score:",r2_score(y_test, y_pred))
print("Mean Absolute Error:",mean_absolute_error(y_test, y_pred))
print("Mean Squared Error:",mean_squared_error(y_test, y_pred))

#hyperparameter tuning
params={
    "criterion": ["squared_error","absolute_error","friedman_mse"],
    "max_depth":[2,4,6,7,8],
    "min_samples_leaf":[2,4,6,7,8],
    "splitter" : ["best","random"]
}

from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(estimator=DecisionTreeRegressor(), param_grid=params,cv=5,n_jobs=-1)
grid_search.fit(X_train, y_train)
y_pred = grid_search.predict(X_test)

print("\nDecision Tree Regressor(After) --- ")
print("R2 Score:",r2_score(y_test, y_pred))
print("Mean Absolute Error:",mean_absolute_error(y_test, y_pred))
print("Mean Squared Error:",mean_squared_error(y_test, y_pred))
print("Best params",grid_search.best_params_)

params_rf={
    "n_estimators":[1,5,10,20],
    "criterion":["squared_error","absolute_error","friedman_mse"],
    "max_depth":[2,4,6,7],
    "min_samples_leaf":[6,7,8],
}
grid_search=GridSearchCV(estimator=RandomForestRegressor(), param_grid=params_rf,cv=5,n_jobs=-1)
grid_search.fit(X_train, y_train)
y_pred = grid_search.predict(X_test)
print("\nRandom Forest Regressor(After) --- ")
print("R2 Score:",r2_score(y_test, y_pred))
print("Mean Absolute Error:",mean_absolute_error(y_test, y_pred))
print("Mean Squared Error:",mean_squared_error(y_test, y_pred))
print("Best params",grid_search.best_params_)
