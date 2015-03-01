import itertools

from util import Enum, merge_dicts

class FeatureExtractor(object):
    FeatureTypes = Enum(['Categorical', 'Numerical'])

    def __init__(self, name, extractor_fn, feature_type=None):
        if feature_type is None:
            feature_type = self.FeatureTypes.Categorical
        self.name = name
        self.feature_type = feature_type
        self._extractor_fn = extractor_fn

    def train(self, parts):
        pass

    def extract_subfeature_names(self, parts):
        if self.feature_type == self.FeatureTypes.Categorical:
            value_set = set(self._extractor_fn(part) for part in parts)
            return [FeatureExtractor.get_categorical_feature_name(
                        self.name, value)
                    for value in value_set]
        else: # feature_type == Numerical
            return [self.name]

    def extract(self, part):
        '''
        Returns a dictionary of subfeature name -> subfeature value. More
        complex feature extractor classes should override this function.
        '''
        feature_value = self._extractor_fn(part)
        if self.feature_type == self.FeatureTypes.Categorical:
            feature_name = FeatureExtractor.get_categorical_feature_name(
                self.name, feature_value)
            return {feature_name: 1.0}
        else: # feature_type == Numerical
            return {self.name: feature_value}

    def extract_all(self, parts):
        return [self.extract(part) for part in parts]

    @staticmethod
    def get_categorical_feature_name(base_name, value):
        return '%s=%s' % (base_name, value)


class KnownValuesFeatureExtractor(FeatureExtractor):
    '''
    This class makes model training more efficient by pre-registering known
    feature values, rather than deriving them from the corpus. The list of
    feature values may include entries that do not in fact appear in the data.

    Using this class implies a categorical feature type.
    '''
    def __init__(self, name, extractor_fn, feature_values):
        super(KnownValuesFeatureExtractor, self).__init__(
              name, extractor_fn, self.FeatureTypes.Categorical)
        self.feature_values = feature_values

    def extract_subfeature_names(self, parts):
        ''' Ignore `parts` and just use known values from initialization. '''
        return [FeatureExtractor.get_categorical_feature_name(
                    self.name, value)
                for value in self.feature_values]

class TrainableFeatureExtractor(FeatureExtractor):
    def __init__(self, name, trainer, feature_extractor_creator,
                  feature_type=FeatureExtractor.FeatureTypes.Categorical):
        self.name = name
        self.feature_type = feature_type
        self.training_results = None
        self._trainer = trainer
        self._feature_extractor_creator = feature_extractor_creator
        self._subfeature_extractors = None

    def train(self, parts):
        self.training_results = self._trainer(parts)
        subfeature_extractors = self._feature_extractor_creator(
            self.training_results)
        self._subfeature_extractors = []
        for subfeature_name, subfeature_extractor in subfeature_extractors:
            full_subfeature_name = '%s:%s' % (self.name, subfeature_name)
            subfeature_extractor = FeatureExtractor(
                full_subfeature_name, subfeature_extractor, self.feature_type)
            self._subfeature_extractors.append(subfeature_extractor)

    def extract_subfeature_names(self, parts):
        assert self._subfeature_extractors is not None, (
            "Cannot retrieve subfeature names before training")
        subfeature_names = [e.extract_subfeature_names(parts)
                            for e in self._subfeature_extractors]
        return list(itertools.chain.from_iterable(subfeature_names))

    def extract(self, part):
        subfeature_values = [e.extract(part)
                             for e in self._subfeature_extractors]
        return merge_dicts(subfeature_values)
