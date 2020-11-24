import numpy as np
from utils import *

with open(inp_file, 'r') as input, open(out_file, 'w') as output:
    csv_write = csv.writer(output)
    for line in csv.reader(input):
        if(isValid(line)):
            csv_write.writerow(line)
for genre in genre_types:
    file_name = 'data/lyrics_' + str(type) + '.csv'
    binClassifierProcess(genre).to_csv(file_name, index=False)
