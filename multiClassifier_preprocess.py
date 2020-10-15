import numpy as np
from utils import *

def PreProcessingMultiClass():
    data = initialProcess()
    for strip_item in strip_items:
        data[text] = data[text].str.strip(strip_item)
    for replace_item in replace_items:
        data[text] = data[text].str.replace(replace_item,'')
    for unwanted_genre in unwanted_genres:
        data = data[data['genre']!=unwanted_genre]
    finalProcess(data).to_csv(final_output, index=False)

with open(inp_file, 'r') as input, open(out_file, 'w') as output:
    csv_write = csv.writer(output)
    for line in csv.reader(input):
        if(isValid(line)):
            csv_write.writerow(line)

PreProcessingMultiClass()
