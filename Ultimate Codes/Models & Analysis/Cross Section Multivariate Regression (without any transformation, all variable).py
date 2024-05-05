# Cross Section Multivariate Regression Model without any transformation (all variables)
# By Siling Song

import pandas as pd
import numpy as np
from sklearn import linear_model
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

def running_cross_section(year):
    df_final_1 = pd.read_csv('/Users/LindaSong/Desktop/cleaned data/merged data 14 sim.csv')
    # saving only the changed year data:
    df_final_1 = df_final_1[df_final_1['LSI changed'] == 1]

    # selecting the years:
    df_final_1 = df_final_1[df_final_1['YEAR'] == year]


    # selecting the variables:
    columns_to_drop = ['Matched_Country','Agricultural Share','Service Share','pl_i', 'pl_c', 'Match_Type', 'LSI changed', 'Source_Line','FDI Net Outflow']
    df_final_1.drop(columns=columns_to_drop, inplace=True)

    # Filtering out rows with empty strings ('') or '..' values
    filtered_df = df_final_1[(df_final_1 != '') & (df_final_1 != '..')].dropna()

    # making sure that all the values are float
    all_columns = filtered_df.columns
    filtered_df[all_columns].astype(float)
    filtered_df = filtered_df.astype(float)


    columns = ['Age dependency ratio', 'IMPORT FROM CHINA','Employment Share Agriculture', 'Employment Share Service','FDI Net Inflow', 'GDP', 'Capital Formation','Labor Force Participation', 'Net Migration','relative price of investment']

    # running the regression:
    # construct the independent and dependent variables
    X_s = filtered_df[columns]
    Y = filtered_df.iloc[:,13]

    # splitting into train and test
    X_train, X_test, Y_train, Y_test = train_test_split(X_s, Y, test_size = 0.2, random_state = 5)

    # running the regression on train data
    Regression = LinearRegression()
    Regression.fit(X=X_train, y=Y_train)

    # saving the coefficients
    Coef = Regression.coef_
    feature_name = Regression.feature_names_in_
    years = [year]*len(feature_name)
    features = {'years': years,
        'feature_name': feature_name,
        'Coef': Coef}
    features_df = pd.DataFrame(features)

    # evaluating the model
    Y_pred = Regression.predict(X_test)
    Y_train_pred = Regression.predict(X_train)
    R2_trained = 1 -np.mean((Y_train_pred-Y_train)**2)/np.var(Y_train)
    print(f'R_square (trained): {R2_trained}')
    R2_test = 1 -np.mean((Y_pred-Y_test)**2)/np.var(Y_test)
    print(f'R_square (test): {R2_test}')

    # saving the R squared values
    year_r2 = [year, year]
    R2 = [R2_trained,R2_test]
    R2_names = ['R2 trained', 'R2 test']
    R2_chart = {'years': year_r2,
        'R2 type': R2_names,
        'R2 value': R2}
    R2_df = pd.DataFrame(R2_chart)

    return features_df, R2_df

# running the function
features = []
R2_chart = []
for year in range(1992,2019): # could change the time periods here
    features_year, R2_chart_year = running_cross_section(year)
    features.append(features_year)
    R2_chart.append(R2_chart_year)


features_combined = pd.concat(features)
R2_combined = pd.concat(R2_chart)
features_combined.reset_index(drop=True, inplace=True)
R2_combined.reset_index(drop=True, inplace=True)

print("Combined Features DataFrame:")
print(features_combined)

print("Combined R-squared DataFrame:")
print(R2_combined)

# plotting feature coefficients:
fig, ax = plt.subplots(figsize=(12, 16))
sns.barplot(x='years', y='Coef', hue='feature_name', data=features_combined, ax=ax)
ax.set_title('Stacked Bar Chart for Coefficients by Year')
ax.set_ylabel('Coefficient')
ax.set_xlabel('Year')
ax.legend(title='Feature Name', loc='upper left', bbox_to_anchor=(1, 1))
plt.show()

# plotting R squared values:
fig, ax = plt.subplots(figsize=(8, 16))
sns.barplot(x='years', y='R2 value', hue='R2 type', data=R2_combined, ax=ax)
ax.set_title('Stacked Bar Chart for R-squared Values by Year')
ax.set_ylabel('R-squared Value')
ax.set_xlabel('Year')
ax.legend(title='R-squared Type', loc='upper left', bbox_to_anchor=(1, 1))
plt.show()







