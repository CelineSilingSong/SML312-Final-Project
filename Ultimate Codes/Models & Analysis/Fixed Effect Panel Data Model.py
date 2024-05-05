# Fixed Effect Panel Data Model Estimate
# by Siling Song

import pandas as pd
import numpy as np
import statsmodels.api as sm
from linearmodels.panel import PooledOLS
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm

# importing the database
df_final_1 = pd.read_csv('/Users/LindaSong/Desktop/cleaned data/please work 2.csv')

# Time frame selection: (can adjust the years to test for different time frame)
df_final_1 = df_final_1[(df_final_1['YEAR'] >= 1994) & (df_final_1['YEAR'] <= 2014)]

# Indexing the dataframe into panel data structure
df_final_1 = df_final_1.set_index(['Matched_Country','YEAR'])

# Extract column names from the DataFrame
all_columns = df_final_1.columns

# Selecting all the features (independent variable & control variables)
exog_x = all_columns[[*range(0, 2), *range(3, 11)]]

# selecting the variables to take log:
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
dep_var = np.log(df_final_1.iloc[:,12]) # taking the log of the target (dependent variable)

# train test split:
X_train, X_test, Y_train, Y_test = train_test_split(exog, dep_var, test_size = 0.2, random_state = 5)

# PanelOLS Modelling:
from linearmodels.panel import PanelOLS
mod_2 = PanelOLS(Y_train, X_train, entity_effects = True)
panel_res = mod_2.fit(cov_type="clustered", cluster_entity=True)
print(panel_res)

# Checking if overfit:
Y_pred = panel_res.predict(X_test)
Y_pred = Y_pred['predictions']
R2 = 1 -np.mean((Y_pred-Y_test)**2)/np.var(Y_test)
print(f'R_square_test: {R2}')

Y_pred_train = panel_res.predict(X_train)
Y_pred_train = Y_pred_train['predictions']
R2_train = 1 -np.mean((Y_pred_train-Y_train)**2)/np.var(Y_train)
print(f'R_square_trained: {R2_train}')

# Plotting the residual:
recid = panel_res.resids
recid_df = recid.to_frame()
mu, std = norm.fit(recid_df)
plt.hist(recid, bins=30, density=True, color='skyblue', edgecolor='black')
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu, std)
plt.plot(x, p, 'k', linewidth=2)
plt.xlabel('Residual')
plt.ylabel('Density')
plt.title('Histogram of residuals')
plt.show()
