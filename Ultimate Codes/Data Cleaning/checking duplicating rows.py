# Checking duplicative rows

import pandas as pd

df = pd.read_csv('/Users/LindaSong/Desktop/cleaned data/merged data 14 sim.csv')

consecutive_duplicates = (df.iloc[:, 0] == df.iloc[:, 0].shift()) & (df.iloc[:, 1] == df.iloc[:, 1].shift())
result = df[consecutive_duplicates]
consolidated_file_path_sim = '/Users/LindaSong/Desktop/cleaned data/duplicative data.csv'

result.to_csv(consolidated_file_path_sim, index = False)