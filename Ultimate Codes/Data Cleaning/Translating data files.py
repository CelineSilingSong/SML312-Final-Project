# translating Import data

import csv
from googletrans import Translator

def translate_csv(input_file, output_file):
    translator = Translator()
    with open(input_file, 'r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        translated_rows = []
        for row in reader:
            translated_row = []
            for cell in row:
                translated_cell = translator.translate(cell, src='zh-cn', dest='en').text
                translated_row.append(translated_cell)
            translated_rows.append(translated_row)
    
    with open(output_file, 'w', encoding='utf-8', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(translated_rows)
        print("translating")

input_file = '/Users/LindaSong/Desktop/cleaned data/Chinese Import 2022 (chinese version).csv'
output_file = '/Users/LindaSong/Desktop/cleaned data/Chinese Import 2022 (English version).csv'
translate_csv(input_file, output_file)