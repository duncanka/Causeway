"""
Some utilities for integrating our pipeline system (particularly the
featurization system) with scikit-learn.
"""

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline, _name_estimators

from nlpypline.pipeline.featurization import Featurizer, FeatureExtractor


class FeaturizingTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, feature_extractors, selected_features_or_name_dict,
                 save_featurized=False, default_to_matrix=True):
        '''
        feature_extractors must be either a list of FeatureExtractors or the
        fully-qualified name of such a list. It must be the latter if the
        transformer is to be pickled.
        '''

        self.featurizer = Featurizer(
            feature_extractors, selected_features_or_name_dict, save_featurized,
            default_to_matrix)
        self.feature_extractors = feature_extractors
        self.selected_features_or_name_dict = selected_features_or_name_dict
        self.save_featurized = save_featurized
        self.default_to_matrix = default_to_matrix

    def fit(self, instances, labels=None):
        self.featurizer.register_features_from_instances(instances)
        return self

    def transform(self, instances):
        return self.featurizer.featurize(instances)

    def num_features(self):
        return len(self.featurizer.feature_name_dictionary)

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['feature_extractors']
        return state


dummy_feature_extractor = FeatureExtractor(
    'dummy', lambda x: (0, FeatureExtractor.FeatureTypes.Numerical))
_dummy_feature_extractors = [dummy_feature_extractor]
def make_mostfreq_featurizing_estimator(estimator_name=None):
    return make_featurizing_estimator(
        DummyClassifier('prior'),
        'nlpypline.skpipeline._dummy_feature_extractors',
        ['dummy'], estimator_name=estimator_name)

def make_featurizing_estimator(
    estimator, feature_extractors, selected_features_or_name_dict,
    estimator_name=None, save_featurized=False, default_to_matrix=True):
    '''
    feature_extractors must be either a list of FeatureExtractors or the
    fully-qualified name of such a list. It must be the latter if the estimator
    is to be pickled.
    '''

    featurizing_stage = FeaturizingTransformer(
        feature_extractors, selected_features_or_name_dict, save_featurized,
        default_to_matrix)
    if estimator_name is not None:
        named = [(estimator_name, estimator)]
    else:
        named = _name_estimators([estimator])
    return Pipeline([(named[0][0] + '_featurizer', featurizing_stage)] + named)
