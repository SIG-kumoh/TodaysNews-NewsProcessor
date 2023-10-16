from hdbscan import HDBSCAN
from sklearn.cluster import MeanShift
import numpy as np


_hdbscan = HDBSCAN(min_cluster_size=3,
                   metric="euclidean",
                   cluster_selection_method='eom')
_mean_shift = MeanShift(bandwidth=0.5)


def clustering(vectors: [np.ndarray]):
    labels = _hdbscan.fit_predict(vectors)
    clusters = _make_cluster(labels, vectors)

    start_label = 0
    max_label = 0
    for cluster in clusters:
        cur_indexes = cluster['indexes']
        cur_vectors = cluster['vectors']
        cur_labels = _mean_shift.fit_predict(cur_vectors)

        for idx, label in zip(cur_indexes, cur_labels):
            label = start_label + label

            labels[idx] = label
            if max_label < label:
                max_label = label
        start_label = max_label + 1

    return labels


def _make_cluster(labels, vectors):
    label_list = labels.tolist()
    list_len = len(list(set(label_list))) - (1 if -1 in label_list else 0)

    clusters = [{'indexes': [], 'vectors': []} for _ in range(list_len)]

    for index, label, vector in zip(range(len(labels)), labels, vectors):
        if label != -1:
            clusters[label]['indexes'].append(index)
            clusters[label]['vectors'].append(vector)

    return clusters
