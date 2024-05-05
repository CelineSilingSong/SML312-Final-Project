# data processing 1

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

csv_file_path = '/Users/LindaSong/Desktop/SML312/Final Project/data/Controlled Variables/Country GDP/Country GDP (world bank)/GDP.csv'

data = {
    'COUNTRY':[],
    'YEAR':[],
    'GDP':[],
} 
df = pd.DataFrame(data)
print(df)

with open(csv_file_path, 'r', encoding='latin-1') as file:
    csv_reader = csv.reader(file, delimiter=',', skipinitialspace=True)
    for x, row in enumerate(csv_reader):
        if row[0] == "Series Name":
            continue
        for y in range(0,31):
            country = row[2]
            year = 1992+y
            gdp = row[4+y]
            new_row = {
                'COUNTRY':country,
                'YEAR':year,
                'GDP':gdp}
            print(new_row)
            df = df._append(new_row, ignore_index = True)

print(df)
consolidated_file_path = '/Users/LindaSong/Desktop/cleaned data/GDP.csv'
df.to_csv(consolidated_file_path, index = False)

