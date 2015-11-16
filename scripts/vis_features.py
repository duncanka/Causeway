import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import vstack, hstack

def select_indices(mat, dim, num):
    num = min(num, mat.shape[dim])
    return np.random.choice(mat.shape[dim], num, replace=False)

def vis_features(features, labels, max_features=500, max_samples=500):
    positive_selector = labels == 1
    negative_selector = labels == 0

    # positive_labels = labels[positive_selector]
    # positive_labels.shape = (positive_labels.shape[0], 1)
    # positives = hstack([features[positive_selector], positive_labels]).tocsr()
    positives = features[positive_selector].tocsr()
    # negative_labels = labels[negative_selector]
    # negative_labels.shape = (negative_labels.shape[0], 1)
    # negatives = hstack([features[negative_selector], negative_labels]).tocsr()
    negatives = features[negative_selector].tocsr()

    feature_indices = select_indices(positives, 1, max_features)
    all_sample_indices = []
    sampled = []
    for mat in [positives, negatives]:
        sample_indices = select_indices(mat, 0, max_samples)
        sampled.append(mat[sample_indices, :][:, feature_indices])
        all_sample_indices.append(sample_indices)
    pos_sampled, neg_sampled = sampled

    all_sampled = vstack([pos_sampled, neg_sampled])
    plt.imshow(all_sampled.toarray(), vmax=3, vmin=-1, aspect='auto',
               interpolation='none')
    # Draw divider
    plt.hlines(min(max_samples, positives.shape[0]) + 0.5, -0.5,
               min(max_features, positives.shape[1]) - 0.5,
               linestyles='dashed', linewidth=2)

    plt.plot()
    plt.show(block=False)
