#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

we assign a positive to each word in trained word2vec model
and rebuild article files with this numbers instead of strings
so comparison between words would be easier in further steps

Attributes:
    store_path (str): where to store processed files
"""
# @Date    : 2017-12-24 11:05:23
# @Author  : Morteza Allahpour (mortezaallahpour@outlook.com)
# @Link    : https://mortza.github.io


from gensim.models import KeyedVectors
import logging
import pandas as pd

logging.basicConfig(filename='log.txt',
                    format='%(asctime)s : %(levelname)s : %(message)s',
                    filemode='a',
                    level=logging.INFO)
store_path = 'processed_files'


def transform_dir(dir_name, vocab_dict, vocab_set):
    """Summary
    
    Args:
        dir_name (TYPE): Description
        vocab_dict (TYPE): Description
    """
    import os

    for file_name in os.listdir(dir_name):
        transform_file(file_name, dir_name, vocab_dict, vocab_set)

    logging.info(f'finish processing directory: {dir_name}')


def transform_file(file_name, dir_name, vocab_dict, vocab_set):
    """replace each word in `vocab_dict` with its index
    then save file in ascii mode in `store_path`/`file_name`
    
    Args:
        file_name (TYPE): Description
        vocab_dict (TYPE): Description
    """
    file = open(dir_name + '/' + file_name, mode='r', encoding='utf-8')

    from nltk.tokenize import RegexpTokenizer
    tokenizer = RegexpTokenizer(r'\w+')

    numbers = []

    for line in file.readlines():
        tokens = tokenizer.tokenize(line)
        for token in tokens:
            if token in vocab_dict:
                numbers.append(vocab_dict[token])
                vocab_set.add(token)
            else:
                numbers.append(-1)
    file.close()

    file = open(store_path + '/' + file_name, mode='w+', encoding='ascii')
    file.write(' '.join([str(i) for i in numbers]))
    file.close()


if __name__ == '__main__':
    w_2_vec = KeyedVectors.load_word2vec_format('FinalModel')
    vocab_dict = {}
    vocab_set = set()
    for word in w_2_vec.vocab:
        # pd_data.append([w_2_vec.vocab[word].index] + w_2_vec.wv[word].tolist())
        vocab_dict[word] = w_2_vec.vocab[word].index

    transform_dir('yjc_ir_political', vocab_dict, vocab_set)
    transform_dir('yjc_ir_social', vocab_dict, vocab_set)
    transform_dir('yjc_ir_sports', vocab_dict, vocab_set)

    series = {}
    pd_data= []
    for word in vocab_set:
        pd_data.append([vocab_dict[word]]+w_2_vec.wv[word].tolist())
        series[word] = vocab_dict[word]

    series = pd.Series(data=series)
    series.to_csv('dictionary.json', mode='w+', encoding='utf-8')
    logging.info(f'saving dictionary.json')

    pd_object = pd.DataFrame(data=pd_data,
                             columns=['tag'] + [f'x{i}' for i in range(100)])

    pd_object.to_csv('features.csv', mode='w+', header=False,
                     encoding='ascii', sep=' ', index=False)
    logging.info(f'saving feature.json, shape: {pd_object.shape}')
