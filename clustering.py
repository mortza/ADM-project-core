import numpy as np
import pandas as pd
import logging

logging.basicConfig(filename='log.txt',
                    format='%(asctime)s : %(levelname)s : %(message)s',
                    filemode='a',
                    level=logging.INFO)

files_path = 'processed_files'
data = pd.read_csv('features.csv', encoding='ascii', sep=' ', header=None,
                   names=['tag'] + [f'x{i}' for i in range(100)], index_col=False,
                   engine='c')
