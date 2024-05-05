# Pooled OLS Model
# By Siling Song

import pandas as pd
import numpy as np
import statsmodels.api as sm
from linearmodels.panel import PooledOLS
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm

# Reading the data file:
df_final_1 = pd.read_csv('/Users/LindaSong/Desktop/cleaned data/please work 2.csv')
df_final_1 = df_final_1[(df_final_1['YEAR'] >= 1994) & (df_final_1['YEAR'] <= 2014)]

# Indexing the data file to panel data structure
df_final_1 = df_final_1.set_index(['Matched_Country','YEAR'])

# Selecting features:
all_columns = df_final_1.columns
exog_x = all_columns[[*range(0, 2), *range(3, 11)]]

# selecting the variables to take logs:
exog_x_ln = exog_x[[*range(0,4), *range(5,8), 9]]
df_final_1[exog_x_ln] = np.log(df_final_1[exog_x_ln])

# selecting the variables to standardize:
columns_to_take_standardize = ['FDI Net Inflow','Net Migration']
data = df_final_1[columns_to_take_standardize]
scaler = StandardScaler()
scaler.fit(data)
data = scaler.transform(data)
df_final_1[columns_to_take_standardize] = data 

# checking & taking care of missing values:
missing_mask = df_final_1.isnull().any(axis=1)
rows_with_missing = df_final_1[missing_mask]
print(rows_with_missing)
rows_with_missing.to_csv('/Users/LindaSong/Desktop/cleaned data/missing rows')

# exog is the features & dep_var is the target:
exog = df_final_1[exog_x]
dep_var = np.log(df_final_1.iloc[:,12])

# train test split:
X_train, X_test, Y_train, Y_test = train_test_split(exog, dep_var, test_size = 0.2, random_state = 5)

# PooledOLS:
mod = PooledOLS(Y_train, X_train)
pooled_res = mod.fit()
print(pooled_res)

# checking if overfit:
Y_pred_train = pooled_res.predict(X_train)
Y_pred_train = Y_pred_train['predictions']
R2_train = 1 -np.mean((Y_pred_train-Y_train)**2)/np.var(Y_train)
Y_pred = pooled_res.predict(X_test)
Y_pred = Y_pred['predictions']
R2_test = 1 -np.mean((Y_pred-Y_test)**2)/np.var(Y_test)
data = {'R squared type': ['train', 'test'],
      'R squared value': [R2_train, R2_test]}
df = pd.DataFrame(data)
print("======================================")
print(df)
print("======================================")

# visualizing coefficients (data scraped from regression results)
coef = [0.0777,
-0.0127,
-0.0344,
-0.0060,
-0.0031,
 0.0043,
-0.0331,
-0.1564,
 0.0310,
-0.0395,]

data = {'feature name': exog_x,
        'coefficient': coef}

df = pd.DataFrame(data)

#Sort the DataFrame by coefficient value
df_sorted = df.sort_values(by='coefficient')

# Plotting
plt.figure(figsize=(10, 6))
plt.barh(df_sorted['feature name'], df_sorted['coefficient'], color='skyblue')
plt.xlabel('Coefficient Value')
plt.ylabel('Feature Name')
plt.title('Coefficients of Features')
plt.grid(True)
plt.show()
