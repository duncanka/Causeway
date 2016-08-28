from __future__ import print_function

from collections import OrderedDict
from gflags import FLAGS
import numpy as np
from sklearn.pipeline import Pipeline
import sys

from util import print_indented


def get_weights_for_classifier(classifier_pipeline):
    classifier = classifier_pipeline.steps[1][1]
    featurizer = classifier_pipeline.steps[0][1].featurizer
    feature_name_dict = featurizer.feature_name_dictionary

    if FLAGS.filter_feature_select_k == -1:
        # All features in feature dictionary are selected.
        feature_indices = range(len(feature_name_dict))
        lr = classifier
    else:
        feature_indices = classifier.named_steps[
            'feature_selection'].get_support().nonzero()[0]
        lr = classifier.named_steps['classification'].classifier

    weights = [(feature_name_dict.ids_to_names[ftnum], lr.coef_[0][i])
               for i, ftnum in enumerate(feature_indices)
               if lr.coef_[0][i] != 0.0]
    weights.sort(key=lambda tup: abs(tup[1])) # sort by weights' absolute values
    return OrderedDict(weights)

def get_general_classifier_weights(pipeline):
    stage = pipeline.stages[3]
    return get_weights_for_classifier(stage.model.global_classifier)

def get_per_connective_weights(pipeline):
    stage = pipeline.stages[3]
    weight_dicts = {}
    for connective, classifier in stage.model.classifiers.iteritems():
        if isinstance(classifier, Pipeline):
            weight_dicts[connective] = None
        else:
            weight_dicts[connective] = get_weights_for_classifier(
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
