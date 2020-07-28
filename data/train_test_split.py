import os 
from os.path import join 
import numpy as np 

DATA_DIR = '.'
queries = open(join(DATA_DIR, "queries.txt")).read().rstrip().split('\n')
queries = {row.split('\t')[0]: row.split('\t')[1] for row in queries}

qs= list(queries.keys())
np.random.shuffle(qs)

train_size = 90000
with open('train.txt', 'w') as f:
    for i in qs[:train_size]:
        f.write(i+'\n')

with open('test.txt', 'w') as f:
    for i in qs[train_size:]:
        f.write(i+'\n')

