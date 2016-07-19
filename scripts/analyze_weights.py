from __future__ import print_function

from collections import OrderedDict
from sklearn.pipeline import Pipeline
import sys

from util import print_indented


def get_weights_for_classifier(classifier_pipeline):
    classifier = classifier_pipeline.steps[1][1]
    featurizer = classifier_pipeline.steps[0][1].featurizer
    support = classifier.named_steps['feature_selection'].get_support()
    lr = classifier.named_steps['classification'].classifier
    weights = [(featurizer.feature_name_dictionary.ids_to_names[ftnum],
                lr.coef_[0][i])
               for i, ftnum in enumerate(support.nonzero()[0])
               if lr.coef_[0][i] != 0.0]
    weights.sort(key=lambda tup: abs(tup[1])) # sort by weights' absolute values
    return OrderedDict(weights)

def get_general_classifier_weights(pipeline):
    stage = pipeline.stages[3]
    return get_weights_for_classifier(stage.model.general_classifier)

def get_per_connective_weights(pipeline):
    stage = pipeline.stages[3]
    weight_dicts = {}
    for connective, classifier in stage.model.classifiers.iteritems():
        if isinstance(classifier, Pipeline):
            weight_dicts[connective] = None
            continue        
        
        weight_dicts[connective] = get_weights_for_classifier(
            dict(classifier.estimators)['per_conn'])

    return weight_dicts

def print_ordered_dict(d, indent=0, file=sys.stdout):
    print_indented(indent, 'OrderedDict({', file=file)
    for key, value in d.items():
        print_indented(indent + 1, key, ': ', value, ',', sep='', file=file)
    print_indented(indent, '})', file=file)

def print_per_connective_weights(pipeline, file=sys.stdout):
    weight_dicts = get_per_connective_weights(pipeline)
    for connective, weights in weight_dicts.iteritems():
        if weights is None:
            print(connective, ': None,', sep='', file=file)
        else:
            print(connective, ':', sep='', file=file, end='')
            print_ordered_dict(weights, 1, file=file)
