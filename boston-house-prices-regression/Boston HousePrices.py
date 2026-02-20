import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

df=pd.read_csv("boston-house-prices.csv")
print(df.head())
print(df.describe())
print(df.info())

columns=['longitude', 'latitude', 'housing_median_age', 'total_rooms',
       'total_bedrooms', 'population', 'households',
       'median_house_value', 'ocean_proximity']

fig,axes=plt.subplots(3,3,figsize=(15,15))
fig.suptitle("Distributions",fontsize=18)

for i,col in enumerate(columns):
    row = i// 3
    col_idx= i % 3
    ax = axes[row,col_idx]
    sns.histplot(data=df,x=col,ax=ax,kde=True)
    ax.set_title(col)
plt.tight_layout()
plt.show()

df["total_bedrooms"].fillna(df["total_bedrooms"].median(), inplace=True)
df = pd.get_dummies(df, columns=["ocean_proximity"], drop_first=True)

print(df.skew(numeric_only=True))

X=df.drop("median_house_value",axis=1)
y=df["median_house_value"]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

"""from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)"""

print(df.skew(numeric_only=True))


from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error


lgbm = LGBMRegressor()
lgbm.fit(X_train, y_train)
y_pred = lgbm.predict(X_test)

print("\n----BEFORE TRANSFORMATION----")
print("LightGBM:")
print("R2 Score: ",r2_score(y_test,y_pred))
print("Mean Squared Error: ",mean_squared_error(y_test,y_pred))
print("Mean Absolute Error: ",mean_absolute_error(y_test,y_pred))

xgboost = XGBRegressor()
xgboost.fit(X_train, y_train)
y_pred = xgboost.predict(X_test)

print("\nXGBoost:")
print("R2 Score: ",r2_score(y_test,y_pred))
print("Mean Squared Error: ",mean_squared_error(y_test,y_pred))
print("Mean Absolute Error: ",mean_absolute_error(y_test,y_pred))

from sklearn.preprocessing import PowerTransformer
power_transformer = PowerTransformer()
X_train_transformed = power_transformer.fit_transform(X_train)
X_test_transformed = power_transformer.transform(X_test)



print("\n---- AFTER TRANSFORMATION ----")

lgbm_transformed=LGBMRegressor()
lgbm_transformed.fit(X_train_transformed, y_train)
y_pred_transformed=lgbm_transformed.predict(X_test_transformed)

print("LightGBM")
print("R2 Score: ",r2_score(y_test,y_pred_transformed))
print("Mean Squared Error: ",mean_squared_error(y_test,y_pred_transformed))
print("Mean Absolute Error: ",mean_absolute_error(y_test,y_pred_transformed))

xgboost_transformed = XGBRegressor()
xgboost_transformed.fit(X_train_transformed, y_train)
y_pred_transformed=xgboost_transformed.predict(X_test_transformed)

print("\nXGBoost:")
print("R2 Score: ",r2_score(y_test,y_pred_transformed))
print("Mean Squared Error: ",mean_squared_error(y_test,y_pred_transformed))
print("Mean Absolute Error: ",mean_absolute_error(y_test,y_pred_transformed))

