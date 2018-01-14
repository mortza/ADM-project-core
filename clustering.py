import numpy as np
import pandas as pd
import logging
from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(filename='log.txt',
                    format='%(asctime)s : %(levelname)s : %(message)s',
                    filemode='a',
                    level=logging.INFO)

files_path = 'processed_files'
data = pd.read_csv('features.csv', encoding='ascii', sep=' ', header=None,
                   names=['tag'] + [f'x{i}' for i in range(100)], index_col=False,
                   engine='c')
# single pass clustering
# need threshold
cols_list = [f'x{i}' for i in range(100)]
wv_s = data.as_matrix(cols_list)
min_sim = 1
max_sim = 0

for (i, _) in enumerate(wv_s):
    print(i)
    for (j, _) in enumerate(wv_s):
        if j == i:
            pass
        sim = cosine_similarity(wv_s[i].reshape(1, -1), wv_s[j].reshape(1, -1))
        max_sim = sim if sim > max_sim else max_sim
        min_sim = sim if sim < min_sim else min_sim

    print(f'''max similarity is : {max_sim}
    min similarity is : {min_sim}''')
    if i == 20:
        break
