import unittest

from pipeline.featurization import (FeatureExtractor, FeaturizationError,
                                    Featurizer)


class FeaturizerTest(unittest.TestCase):
    def setUp(self):
        self.identity_extractor = FeatureExtractor(
            'identity', lambda x: x, FeatureExtractor.FeatureTypes.Categorical)
        self.add1_extractor = FeatureExtractor(
            'add1', lambda x: x + 1, FeatureExtractor.FeatureTypes.Categorical)
        self.featurizer = Featurizer(
            [self.identity_extractor, self.add1_extractor],
            ['identity', 'add1', 'identity:add1'])

    def test_subfeature_names(self):
        self.featurizer.register_features_from_instances([1, 2])
        subfeature_names = set(
            self.featurizer.feature_name_dictionary.names_to_ids.keys())
        correct = set(['identity=1', 'identity=2', 'add1=2', 'add1=3',
                       'identity=1:add1=2', 'identity=2:add1=3',
                       'identity=2:add1=2', 'identity=1:add1=3'])
        self.assertSetEqual(correct, subfeature_names)

    def test_complains_for_invalid_feature_names(self):
        def set_invalid_feature():
            self.featurizer = Featurizer([self.identity_extractor], ['add1'])
        self.assertRaises(FeaturizationError, set_invalid_feature)

        def set_invalid_combination():
            self.featurizer = Featurizer([self.identity_extractor],
                                         ['identity:add1'])
        self.assertRaises(FeaturizationError, set_invalid_combination)
