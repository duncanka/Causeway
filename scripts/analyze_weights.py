from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
from sklearn.pipeline import Pipeline
import sys

from causeway.candidate_filter import get_weights_for_lr_classifier
from nlpypline.util import print_indented


def get_global_classifier_weights(pipeline):
    stage = pipeline.stages[3]
    return get_weights_for_lr_classifier(stage.model.global_classifier)

def get_per_connective_weights(pipeline):
    stage = pipeline.stages[3]
    weight_dicts = {}
    for connective, classifier in stage.model.classifiers.iteritems():
        if isinstance(classifier, Pipeline):
            weight_dicts[connective] = None
        else:
            weight_dicts[connective] = get_weights_for_lr_classifier(
                dict(classifier.estimators)['perconn'])

    return weight_dicts

def get_inter_classifier_weights(pipeline):
    stage = pipeline.stages[3]
    weights_by_connective = {}
    for connective, classifier in stage.model.classifiers.iteritems():
        if isinstance(classifier, Pipeline):
            weights_by_connective[connective] = np.array([0, 1, 0])
        else:
            weights_by_connective[connective] = classifier.weights

    return weights_by_connective

def print_ordered_dict(d, indent=0, file=sys.stdout):
    print_indented(indent, 'OrderedDict({', file=file)
    for key, value in d.items():
        print_indented(indent + 1, key, ': ', value, ',', sep='', file=file)
    print_indented(indent, '})', file=file)

def print_per_connective_weights(pipeline, file=sys.stdout):
    weight_dicts = get_per_connective_weights(pipeline)
    for connective, weights in sorted(weight_dicts.iteritems()):
        if weights is None:
            print(connective, ': None,', sep='', file=file)
        else:
            print(connective, ':', sep='', file=file)
            print_ordered_dict(weights, 1, file=file)

def plot_hist(d, key, color='blue', show=True):
    plt.hist(d[key], bins='auto', color=color, alpha=0.7)
    plt.title(key)
    if show:
        plt.show()
