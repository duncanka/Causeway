import unittest

from nlpypline.pipeline.featurization import (FeatureExtractor, FeaturizationError,
                                    Featurizer, NestedFeatureExtractor)


class FeaturizationTest(unittest.TestCase):
    def setUp(self):
        self.identity_extractor = FeatureExtractor(
            'identity', lambda x: x, FeatureExtractor.FeatureTypes.Categorical)
        self.add1_extractor = FeatureExtractor(
            'add1', lambda x: x + 1, FeatureExtractor.FeatureTypes.Categorical)
        self.featurizer = Featurizer(
            [self.identity_extractor, self.add1_extractor],
            ['identity', 'add1', 'identity:add1'])


class NestedFeatureExtractorTest(FeaturizationTest):
    def setUp(self):
        FeaturizationTest.setUp(self)
        self.nested_extractor = NestedFeatureExtractor(
            'nested', [self.identity_extractor, self.add1_extractor])

    def test_nested_extractor_subfeatures(self):
        subfeature_names = self.nested_extractor.extract_subfeature_names([1,
                                                                           2])
        self.assertEqual(subfeature_names,
                         ['nested_identity=1', 'nested_identity=2',
                          'nested_add1=2', 'nested_add1=3'])

    def test_nested_extractor_extraction(self):
        extracted = self.nested_extractor.extract(1)
        self.assertEqual(extracted, {key: 1.0 for key in ['nested_identity=1',
                                                          'nested_add1=2']})
        extracted = self.nested_extractor.extract(2)
        self.assertEqual(extracted, {key: 1.0 for key in ['nested_identity=2',
                                                          'nested_add1=3']})


class FeaturizerTest(FeaturizationTest):
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
