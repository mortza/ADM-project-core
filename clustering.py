import numpy as np
import pandas as pd
import logging
from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(filename='log.txt',
                    format='%(asctime)s : %(levelname)s : %(message)s',
                    filemode='a',
                    level=logging.WARNING)

files_path = 'processed_files'

data = pd.read_csv('features.csv', encoding='ascii', sep=' ', header=None,
                   names=['tag'] + [f'x{i}' for i in range(100)], index_col=False,
                   engine='c')

logging.info(f'features.csv loaded, pd DataFrame shape: {data.shape}')
cols_list = [f'x{i}' for i in range(100)]
wv_s = data.as_matrix(cols_list)

# single pass clustering
# threshold set to 0.5
# as it said, word orders affect performance

single_pass = False

if single_pass:
    t = 0.5
    centers = {1: wv_s[0]}
    centers_count = {1: 1}
    centers_num = 1

    for (i, _) in enumerate(wv_s):
        if i == 1:
            continue
        if i % 1000 == 0:
            logging.info(f'single pass clustering: iteration= {i}, number of clusters= {centers_num}')
        # find similarity to each center
        max_sim = -1
        max_sim_c = -1  # invalid index
        for key in centers.keys():
            sim = cosine_similarity(wv_s[i].reshape(1, -1), centers[key].reshape(1, -1))[0][0]
            if sim > max_sim:
                max_sim = sim
                max_sim_c = key

        # check for threshold
        if max_sim > t:
            # assign to corresponding cluster
            centers[max_sim_c] = (centers_count[max_sim_c] * centers[max_sim_c] + wv_s[i]) \
                                 / (centers_count[max_sim_c] + 1)
            centers_count[max_sim_c] = centers_count[max_sim_c] + 1
        else:
            # create new cluster
            centers_num = centers_num + 1
            centers[centers_num] = wv_s[i]
            centers_count[centers_num] = 1

    logging.info('dumping to file...')
    import pickle

    dump_file = open('single_pass_cluster_dump', mode='wb')
    pickle.dump(centers, dump_file)
    dump_file.close()
    logging.info('persisted to single_pass_cluster_dump in PWD')
    logging.info(f'single pass clustering: number of clusters= {centers_num}')
else:
    import pickle

    initial_centers = open('single_pass_cluster_dump', mode='rb')
    initial_centers = pickle.load(initial_centers)
    initial_centers = np.array([val for val in initial_centers.values()])
    from sklearn.cluster import KMeans

    kmns = KMeans(n_clusters=len(initial_centers), init=initial_centers,
                  max_iter=5, precompute_distances='auto')
    lbls = kmns.fit_predict(wv_s)
    data['label'] = lbls
    grouped_object = data.groupby('label').apply(lambda arg: arg.tag.values)

    final_clusters = open('final_clusters.txt', mode='w+', encoding='ascii')
    for key in grouped_object.keys():
        str_rep = ' '.join([str(i) for i in grouped_object[key]])
        final_clusters.write(str_rep)
        final_clusters.write('\n')
    final_clusters.close()
