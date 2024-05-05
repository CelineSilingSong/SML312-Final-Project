# processing Labor Share of Income 1992 to 2021

import os
import tabula
import csv
import pandas as pd
import re
from pdf2image import convert_from_path
import PyPDF2
import tabula

os.environ['JAVA_HOME'] = '/Library/Java/JavaVirtualMachines/temurin-21.jdk/Contents/Home'

print("JAVA_HOME:", os.environ.get("JAVA_HOME"))
print("PATH:", os.environ.get("PATH"))

trade_folder = '/Users/LindaSong/Desktop/SML312/Final Project/data/Labor share of income by country'

data = {
    'COUNTRY':[],
    'YEAR':[],
    'Labor Share of Income':[],
    'LSI changed':[]
} 
df = pd.DataFrame(data)
print(df)

previous_LSI = 0

for foldername, _, filenames in os.walk(trade_folder):
    for filename in filenames:
        if filename.endswith('.csv'):
            csv_file_path = os.path.join(foldername,filename)
            with open(csv_file_path, 'r', encoding='latin-1') as file:
                csv_reader = csv.reader(file, delimiter=',', skipinitialspace=True)
                next(csv_reader) 
                for x, row in enumerate(csv_reader):
                    if (row[0] == "Reporter Name"):
                        continue
                    else:
                        print(row)
                        country = filename
                        year = float(row[0].split('-')[0])
                        print(year)
                        current_LSI = row[1]
                        if (current_LSI == previous_LSI):
                            changed = 0
                        else:
                            changed = 1
                            previous_LSI = current_LSI
                        new_row = {
                            'COUNTRY':country,
                            'YEAR':year,
                            'Labor Share of Income':current_LSI,
                            'LSI changed':changed}
                            
                        print(new_row)
                        df = df._append(new_row, ignore_index = True)
                        continue

print(df)
consolidated_file_path = '/Users/LindaSong/Desktop/cleaned data/Labor Share of income.csv'
df.to_csv(consolidated_file_path, index = False)
