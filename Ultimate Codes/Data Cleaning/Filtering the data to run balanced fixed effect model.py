# Filtering the data to run balanced fixed effect model

import pandas as pd

# importing data to clean:
df_final = pd.read_csv('/Users/LindaSong/Desktop/cleaned data/merged data 14 sim.csv')

# Dropping some of the variables
columns_to_drop = ['Agricultural Share','Service Share','pl_i', 'pl_c', 'Match_Type', 'Source_Line','FDI Net Outflow']
df_final.drop(columns=columns_to_drop, inplace=True)

# Filtering out rows with empty strings ('') or '..' values
filtered_df = df_final[(df_final != '') & (df_final != '..')].dropna()

# making sure that all the values are float
all_columns = filtered_df.columns
filtered_df[all_columns[2:15]].astype(float)

# Group by the country column and count occurrences of 0 in the 'LSI changed' column
no_LSI_change_counts = filtered_df.groupby('Matched_Country')['LSI changed'].apply(lambda x: (x == 0).sum())

# Filter out the groups (countries) where the count exceeds 3
countries_to_keep = no_LSI_change_counts[no_LSI_change_counts <= 3].index

# Drop the rows corresponding to the countries to be dropped
filtered_df = filtered_df[filtered_df['Matched_Country'].isin(countries_to_keep)]
all_columns = filtered_df.columns

# dropping some of the variables
columns_to_drop = ['LSI changed']
filtered_df.drop(columns=columns_to_drop, inplace=True)

# making sure that only countries that have full time length (between year x and year y) are kept in place
filtered_df = filtered_df[filtered_df['YEAR']>=1994] # year x
filtered_df = filtered_df[filtered_df['YEAR']<=2019] # year y
country_year_counts = filtered_df.groupby('Matched_Country')['YEAR'].nunique()
filtered_countries = country_year_counts[country_year_counts == 26].index.tolist()
filtered_df = filtered_df[filtered_df['Matched_Country'].isin(filtered_countries)]

# produce a filtered csv file
filtered_df.to_csv('/Users/LindaSong/Desktop/cleaned data/please work 2.csv', index= False)