# Preprocessing chinese data to translate to English:
import os
import tabula
import csv
import pandas as pd
import re
import PyPDF2

os.environ['JAVA_HOME'] = '/Library/Java/JavaVirtualMachines/temurin-21.jdk/Contents/Home'

print("JAVA_HOME:", os.environ.get("JAVA_HOME"))
print("PATH:", os.environ.get("PATH"))

csv_file_path = '/Users/LindaSong/Desktop/SML312/Final Project/data/Controlled Variables/foreign direct investment/FDI net outflow/FDI net outflow.csv'

data = {
    'COUNTRY':[],
    'YEAR':[],
    'CHINESE IMPORT':[],
} 
df = pd.DataFrame(data)
print(df)

def preprocess_country_name (input_string):
    # Substrings to check for
    substring1 = "中国向"
    substring2 = "出口总额(万美元)"

    # Check if both substrings are present
    if substring1 in input_string and substring2 in input_string:
        start_index = input_string.find(substring1) + len(substring1)
        end_index = input_string.find(substring2)
        edited_string = input_string[start_index:end_index]
    else:
        edited_string = input_string

    print("Edited string:", edited_string)
    return edited_string

trade_folder = "/Users/LindaSong/Desktop/SML312/Final Project/data/Trade data with China/2022"

# loop through all the folders in the root director
for foldername, _, filenames in os.walk(trade_folder):
    for filename in filenames:
        if filename.endswith('.csv'):
            csv_file_path = os.path.join(foldername,filename)
            with open(csv_file_path, 'r', encoding='gb2312') as file:
                csv_reader = csv.reader(file, delimiter=',', skipinitialspace=True)
                for x, row in enumerate(csv_reader):
                    if row[0].__contains__("数据库") :
                        continue
                    if row[0].__contains__("时间") :
                        continue
                    if row[0].__contains__("指标") :
                        continue
                    if row[0].__contains__("中华人民共和国") :
                        continue
                    if len(row) < 2:
                        print(row)
                        continue
                    print(row)
                    country = preprocess_country_name(row[0])
                    print(country)
                    year = 2022
                    chinese_import = float(row[1])*10
                    new_row = {
                        'COUNTRY':country,
                        'YEAR':year,
                        'CHINESE IMPORT':chinese_import}
                    print(new_row)
                    df = df._append(new_row, ignore_index = True)

consolidated_file_path = '/Users/LindaSong/Desktop/cleaned data/Chinese Import 2022 (chinese version).csv'
df.to_csv(consolidated_file_path, index = False)