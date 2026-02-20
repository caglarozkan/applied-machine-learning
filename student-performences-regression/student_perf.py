import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv("Student_Performance.csv")
print(df.head())
print(df.info())

print(df.isnull().sum())

"""import math
def plot_all_histograms(df, title_prefix=""):
    num_cols = df.select_dtypes(include=[np.number]).columns
    n_cols = 2
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

print(plot_all_histograms(df))"""

print(df["Extracurricular Activities"].value_counts())

df["Extracurricular Activities"] = df["Extracurricular Activities"].map({"No": 0, "Yes": 1})
print(df["Extracurricular Activities"].value_counts())

X=df.drop("Performance Index",axis=1)
y=df["Performance Index"]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)

from sklearn.linear_model import LinearRegression
linear_reg=LinearRegression()
linear_reg.fit(X_train,y_train)
prediction=linear_reg.predict(X_test)

sns.scatterplot(x=y_test,y=prediction)
plt.show()

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, root_mean_squared_error
print("\nR2 Score:",r2_score(y_test,prediction))
print("Mean Squared Error:",mean_squared_error(y_test,prediction))
print("Mean Absolute Error:",mean_absolute_error(y_test,prediction))

#Ridge / Lasso kullanacaksan scaling şarttır.
from sklearn.linear_model import Ridge

ridge=Ridge()
ridge.fit(X_train,y_train)
prediction=ridge.predict(X_test)

print("\n--- RIDGE ---")
print("R2 score:",r2_score(y_test,prediction))
print("Mean Squared Error:",mean_squared_error(y_test,prediction))
print("Mean Absolute Error:",mean_absolute_error(y_test,prediction))
print("RMSE",root_mean_squared_error(y_test,prediction))


from sklearn.linear_model import Lasso
lasso=Lasso()
lasso.fit(X_train,y_train)
prediction=lasso.predict(X_test)
print("\n--- LASSO ---")
print("R2 score:",r2_score(y_test,prediction))
print("Mean Squared Error:",mean_squared_error(y_test,prediction))
print("Mean Absolute Error:",mean_absolute_error(y_test,prediction))
print("RMSE",root_mean_squared_error(y_test,prediction))

params_ridge={
    "alpha":[0.1,0.3,0.5]
}

param_lasso={
    "alpha" : [0.1,0.3,0.5]
}
#hyperparameter tuning
from sklearn.model_selection import GridSearchCV
grid_search=GridSearchCV(
    estimator=Ridge(),
    param_grid=params_ridge,
    n_jobs=1,
    cv=5
)
grid_search.fit(X_train,y_train)
prediction=grid_search.predict(X_test)

print("\n --- RIDGE(AFTER) ---")
print("R2 score:",r2_score(y_test,prediction))
print("Mean Squared Error:",mean_squared_error(y_test,prediction))
print("Mean Absolute Error:",mean_absolute_error(y_test,prediction))
print("RMSE",root_mean_squared_error(y_test,prediction))

grid_search=GridSearchCV(
    estimator=Lasso(),
    param_grid=param_lasso,
    n_jobs=1,
    cv=5
)
grid_search.fit(X_train,y_train)
prediction=grid_search.predict(X_test)
print("\n--- LASSO(After) ---")
print("R2 score:",r2_score(y_test,prediction))
print("Mean Squared Error:",mean_squared_error(y_test,prediction))
print("Mean Absolute Error:",mean_absolute_error(y_test,prediction))
print("RMSE",root_mean_squared_error(y_test,prediction))

"""coef_df = pd.DataFrame({
    "Feature": X.columns,
    "Linear": linear_reg.coef_,
    "Ridge": ridge.coef_,
    "Lasso": lasso.coef_
})

print("\n",coef_df)"""