# Clearning (Second to last)

import pandas as pd

df_final = pd.read_csv("/Users/LindaSong/Desktop/cleaned data/merged data 14 sim.csv")

df_final = df_final[df_final['Matched_Country']!='Central African Republic']
df_final = df_final[df_final['Matched_Country']!='Eritrea']
df_final = df_final[df_final['Matched_Country']!='East Asia & Pacific']
df_final = df_final[df_final['Matched_Country']!='Fragile and conflict affected situations']
df_final = df_final[df_final['Matched_Country']!='Montenegro']
df_final = df_final[df_final['Matched_Country']!='']

df_final.to_csv("/Users/LindaSong/Desktop/cleaned data/merged data 14 sim.csv",index = False)