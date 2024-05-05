# merging all the data:


from rapidfuzz import fuzz
from rapidfuzz import process
import pandas as pd

# reading all of the csv files
df_age_dep = pd.read_csv('/Users/LindaSong/Desktop/cleaned data/Age dependency ratio.csv')
df_agr_sha = pd.read_csv('/Users/LindaSong/Desktop/cleaned data/Agricultural share.csv')
df_chi_imp = pd.read_csv('/Users/LindaSong/Desktop/cleaned data/Chinese Import all.csv')
df_dom_cre = pd.read_csv('/Users/LindaSong/Desktop/cleaned data/Domestic Credit Private.csv')
df_emp_agr = pd.read_csv('/Users/LindaSong/Desktop/cleaned data/Employment Share Agriculture.csv')
df_emp_ser = pd.read_csv('/Users/LindaSong/Desktop/cleaned data/Employment Share Service.csv')
df_FDI_out = pd.read_csv('/Users/LindaSong/Desktop/cleaned data/FDI net outflow.csv')
df_FDI_in = pd.read_csv('/Users/LindaSong/Desktop/cleaned data/FDI netinflow.csv')
df_GDP = pd.read_csv('/Users/LindaSong/Desktop/cleaned data/GDP.csv')
df_cap_for = pd.read_csv('/Users/LindaSong/Desktop/cleaned data/Gross fixed capital formation.csv')
df_lab_par = pd.read_csv('/Users/LindaSong/Desktop/cleaned data/Labor Force Participation.csv')
df_lab_sha = pd.read_csv('/Users/LindaSong/Desktop/cleaned data/Labor Share of income.csv')
df_net_mig = pd.read_csv('/Users/LindaSong/Desktop/cleaned data/Net Migration.csv')
df_rel_inv = pd.read_csv('/Users/LindaSong/Desktop/cleaned data/relative price of investment.csv')
df_ser_sha = pd.read_csv('/Users/LindaSong/Desktop/cleaned data/Service share.csv')
df_tot_imp = pd.read_csv('/Users/LindaSong/Desktop/cleaned data/Total Import.csv')

dfs = [df_age_dep,df_agr_sha,df_chi_imp,df_dom_cre,df_emp_agr,df_emp_ser,df_FDI_out,df_FDI_in,df_GDP,df_cap_for,
       df_lab_par,df_net_mig,df_rel_inv,df_ser_sha,df_tot_imp,df_lab_sha]

# Function to perform fuzzy matching and merge dataframes
def get_most_similar_country(row, choices):
    country_name = row['COUNTRY']
    match_result = process.extractOne(country_name, choices)
    if match_result is None:
        return 'No Match', 'No Match', 0
    else:
        matched_country, score, index = process.extractOne(country_name, choices)
        if score >= 100:
            return matched_country, 'Exact Match', index
        elif score >= 75:
            print("!!!!!!!!!!!!!!!!!!!!")
            print(f"country name is {country_name}")
            print(match_result)
            print("!!!!!!!!!!!!!!!!!!!!!!")
            return matched_country, 'Similar Match', index
        else:
            print(f"country name is {country_name}")
            print(match_result)
            return 'No Match', 'No Match', 0

def fuzzy_merge(df1, df2):
    if 'COUNTRY' in df1.columns:
        df1 = df1.rename(columns={'COUNTRY': 'Matched_Country'})
    df2[['Matched_Country', 'Match_Type', 'Source_Line']] = df2.apply(lambda row: get_most_similar_country(row, df1['Matched_Country']), axis=1, result_type='expand')
    df2 = df2.drop('COUNTRY', axis =1)
    column_list_df2 = list(df2.columns)
    place_holder = column_list_df2[1]
    column_list_df2[1] = column_list_df2[2]
    column_list_df2[2] = place_holder
    df2 = df2.reindex(columns = column_list_df2)
    merged_df = pd.merge(df1, df2, on=['YEAR', 'Matched_Country'], how='outer')
    merged_df = merged_df[merged_df['Matched_Country'] != 'No Match']
    merged_df_sim = merged_df.drop(columns=['Match_Type', 'Source_Line'], axis=1)

    return merged_df, merged_df_sim

# Merge each dataframe with the first one (dfs[0])
merged_df_sim=dfs[0]
for x, df in enumerate(dfs[1:]):
    merged_df, merged_df_sim = fuzzy_merge(merged_df_sim,df)
    consolidated_file_path = f'/Users/LindaSong/Desktop/cleaned data/merged data {x}.csv'
    merged_df.to_csv(consolidated_file_path, index = False)
    consolidated_file_path_sim = f'/Users/LindaSong/Desktop/cleaned data/merged data {x} sim.csv'
    merged_df.to_csv(consolidated_file_path_sim, index = False)




