from __future__ import print_function
from collections import OrderedDict

def get_weights_for_classifier(classifier_pipeline):
    featurizer = classifier_pipeline.steps[0][1].featurizer
    classifier = classifier_pipeline.steps[1][1]
    support = classifier.named_steps['feature_selection'].get_support()
    lr = classifier.named_steps['classification'].classifier
    weights = [(featurizer.feature_name_dictionary.ids_to_names[ftnum],
                lr.coef_[0][i])
               for i, ftnum in enumerate(support.nonzero()[0])]
    weights.sort(key=lambda tup: abs(tup[1])) # sort by weights' absolute values
    return OrderedDict(weights)

def get_general_classifier_weights(pipeline):
    stage = pipeline.stages[3]
    return get_weights_for_classifier(stage.model.general_classifier)

def get_per_connective_weights(pipeline):
    stage = pipeline.stages[3]
    return {connective: get_weights_for_classifier(classifier)
            for connective, classifier in stage.model.classifiers.iteritems()}

def print_ordered_dict(d):
    print('OrderedDict({')
    for key, value in d.items():
        print('  ', key, ': ', value, sep='')
    print('})')
