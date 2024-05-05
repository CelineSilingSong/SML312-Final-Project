# Checking if fuzzy match matched wrongly

from rapidfuzz import fuzz
from rapidfuzz import process
import pandas as pd

df_age_dep = pd.read_csv('/Users/LindaSong/Desktop/cleaned data/Age dependency ratio.csv')
df_rel_inv = pd.read_csv('/Users/LindaSong/Desktop/cleaned data/relative price of investment.csv')
df_lab_sha = pd.read_csv('/Users/LindaSong/Desktop/cleaned data/Labor Share of income.csv')
df_chi_imp = pd.read_csv('/Users/LindaSong/Desktop/cleaned data/Chinese Import.csv')
df_chi_imp_2022 = pd.read_csv('/Users/LindaSong/Desktop/cleaned data/Chinese Import 2022 (English version).csv')

df_age_dep = df_age_dep[df_age_dep['YEAR'] == 1992]
df_rel_inv = df_rel_inv[df_rel_inv['YEAR'] == 1992]
df_lab_sha = df_lab_sha[df_lab_sha['YEAR'] == 1992]
df_chi_imp = df_chi_imp[df_chi_imp['YEAR'] == 1992]

def get_most_similar_country(row, choices):
    country_name = row['COUNTRY']
    match_result = process.extractOne(country_name, choices)
    if match_result is None:
        return 'No Match', 'No Match', score, 0
    else:
        matched_country, score, index = process.extractOne(country_name, choices)
        if score >= 100:
            return country_name, matched_country, 'Exact Match', score, index
        elif score >= 75:
            print("!!!!!!!!!!!!!!!!!!!!")
            print(f"country name is {country_name}")
            print(match_result)
            print("!!!!!!!!!!!!!!!!!!!!!!")
            return country_name, matched_country, 'Similar Match', score, index
        else:
            print(f"country name is {country_name}")
            print(match_result)
            return country_name, matched_country, 'No Match', score, 0
        
def fuzzy_matched_pairs(df1, df2, df_country_pairs):
    df_country_pairs[['COUNTRY', 'Matched Country', 'Match_Type','Score','Source_Line']] = df2.apply(lambda row: get_most_similar_country(row, df1['COUNTRY']), axis=1, result_type='expand')

    return df_country_pairs

df_country_pairs = pd.DataFrame()
df_country_pairs = fuzzy_matched_pairs(df_age_dep,df_chi_imp_2022,df_country_pairs)

df_country_pairs.to_csv('/Users/LindaSong/Desktop/cleaned data/Country Pair.csv')