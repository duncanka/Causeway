from gflags import DEFINE_string, FLAGS, DuplicateFlagError
import itertools
import logging
import numpy as np
from scipy.sparse import lil_matrix
import time

from util import Enum, merge_dicts, NameDictionary

try:
    DEFINE_string('conjoined_feature_sep', ':',
                  'Separator character to use between conjoined feature names.'
                  ' This character can still be used in conjoined feature names'
                  ' by doubling it (e.g., "f1=:::f2=x").')
    DEFINE_string('subfeature_sep', '_',
                  'Separator character to use between trained subfeatures.')
except DuplicateFlagError as e:
    logging.warn('Ignoring redefinition of flag %s' % e.flagname)


class FeaturizationError(Exception):
    pass


class FeatureExtractor(object):
    FeatureTypes = Enum(['Categorical', 'Numerical']) # Numerical includes bool

    @staticmethod
    def escape_conjoined_name(feature_name):
        sep = FLAGS.conjoined_feature_sep
        return feature_name.replace(sep, sep * 2)

    @staticmethod
    def separate_conjoined_feature_names(conjoined_names):
        ''' Returns unescaped split feature names. '''
        sep = FLAGS.conjoined_feature_sep
        double_sep = sep * 2
        conjoined_names = conjoined_names.replace(double_sep, '\0')
        return [name.replace('\0', sep)
                for name in conjoined_names.split(sep)]

    @staticmethod
    def conjoin_feature_names(feature_names):
        return FLAGS.conjoined_feature_sep.join(feature_names)

    def __init__(self, name, extractor_fn, feature_type=None):
        if feature_type is None:
            feature_type = self.FeatureTypes.Categorical
        self.name = name
        self.feature_type = feature_type
        self._extractor_fn = extractor_fn

    def train(self, instances):
        pass

    def extract_subfeature_names(self, instances):
        if self.feature_type == self.FeatureTypes.Categorical:
            values_set = set(self._extractor_fn(part) for part in instances)
            return [FeatureExtractor._get_categorical_feature_name(
                        self.name, value)
                    for value in values_set]
        else: # feature_type == Numerical
            return [self.name]

    def extract(self, part):
        '''
        Returns a dictionary of subfeature name -> subfeature value. More
        complex feature extractor classes should override this function.
        '''
        feature_value = self._extractor_fn(part)
        if self.feature_type == self.FeatureTypes.Categorical:
            feature_name = FeatureExtractor._get_categorical_feature_name(
                self.name, feature_value)
            return {feature_name: 1.0}
        else: # feature_type == Numerical
            return {self.name: feature_value}

    def extract_all(self, parts):
        return [self.extract(part) for part in parts]

    @staticmethod
    def _get_categorical_feature_name(base_name, value):
        return '%s=%s' % (base_name, value)

    def __repr__(self):
        return 'Feature extractor: %s' % self.name


class Featurizer(object):
    class ConjoinedFeatureExtractor(FeatureExtractor):
        '''
        A Featurizer allows combining features. This class provides the
        necessary functionality of combining the FeatureExtractors' outputs.
        '''

        def __init__(self, name, extractors):
            self.name = name
            self.feature_type = self.FeatureTypes.Categorical
            for extractor in extractors:
                if extractor.feature_type != self.FeatureTypes.Categorical:
                    raise FeaturizationError(
                        "Only categorical features can be conjoined (attempted"
                        " to conjoin %s)" % [e.name for e in extractors])
            self._extractors = extractors

        def extract_subfeature_names(self, instances):
            subfeature_name_components = [
                extractor.extract_subfeature_names(instances)
                for extractor in self._extractors]
            return [self.conjoin_feature_names(components) for components in
                    itertools.product(*subfeature_name_components)]

        def train(self, instances):
            for extractor in self._extractors:
                extractor.train(instances)

        def extract(self, instance):
            # Sub-extractors may produce entire dictionaries of feature values
            # (e.g., for set-valued feature extractors). We need to produce one
            # conjoined feature for every element of the Cartesian product of
            # these dictionaries.
            # Note that the extractor results come out in the same order as the
            # features were initially specified, so we can safely construct the
            # conjoined name by just joining these subfeature names.
            extractor_results = [extractor.extract(instance).keys()
                                 for extractor in self._extractors]
            # Separator characters must be escaped in conjoined names.
            escaped = [[FeatureExtractor.escape_conjoined_name(name)
                        for name in extractor_result]
                       for extractor_result in extractor_results]
            cartesian_product = itertools.product(*escaped)
            sep = FLAGS.conjoined_feature_sep
            return {sep.join(subfeature_names): 1.0
                    for subfeature_names in cartesian_product}


    def __init__(self, feature_extractors, selected_features_or_name_dict,
                 instance_filter=None, save_featurized=False):
        """
        feature_extractors is a list of
            `pipeline.featurization.FeatureExtractor` objects.
        selected_features_or_name_dict must be one of the following:
            - a list of names of features to extract. Names may be combinations
              of feature names, separated by FLAGS.conjoined_feature_sep. (This
              character should be escaped in the names of any conjoined features
              containing it, or the conjoined features may not work properly.)
            - a NameDictionary that should be used for this Featurizer. This
              implicitly encodes the selected features.
        instance_filter is a filter function that takes an instance and returns
            True iff it should be featurized. Instances that are filtered out
            will be featurized as all zeros.
        save_featurized indicates whether to store features and labels
            properties after featurization. Useful for debugging/development.
        """
        self.all_feature_extractors = feature_extractors
        self.feature_training_data = []
        self.save_featurized = save_featurized
        self.feature_extractors = [] # for IDE's information
        self._instance_filter = instance_filter

        if isinstance(selected_features_or_name_dict, NameDictionary):
            self.feature_name_dictionary = selected_features_or_name_dict
            selected_features = self.get_selected_features(
                selected_features_or_name_dict)
        else:
            self.feature_name_dictionary = NameDictionary()
            selected_features = selected_features_or_name_dict
        self._initialize_feature_extractors(selected_features)

    def reset(self):
        self.feature_name_dictionary.clear()
        self.feature_training_data = []

    def register_feature_names(self, feature_names):
        for feature_name in feature_names:
            self.feature_name_dictionary.insert(feature_name)

    def register_features_from_instances(self, instances):
        # Build feature name dictionary. (Unfortunately, this means we
        # featurize many things twice, but I can't think of a cleverer way to
        # discover the possible values of a feature.)
        for extractor in self.feature_extractors:
            self.feature_training_data.append(extractor.train(instances))
            subfeature_names = extractor.extract_subfeature_names(instances)
            self.register_feature_names(subfeature_names)

    def featurize(self, instances):
        logging.debug('Featurizing...')
        start_time = time.time()

        features = lil_matrix(
            (len(instances), len(self.feature_name_dictionary)),
            dtype=np.float32) # TODO: Make this configurable?

        for instance_index, instance in enumerate(instances):
            if self._instance_filter and not self._instance_filter(instance):
                continue

            # TODO: make featurization not re-run combined feature extractors.

            for extractor in self.feature_extractors:
                instance_subfeature_values = extractor.extract(instance)
                for subfeature_name, subfeature_value in (
                    instance_subfeature_values.iteritems()):

                    if subfeature_value == 0:
                        continue # Don't bother setting 0's in a sparse matrix.
                    try:
                        feature_index = self.feature_name_dictionary[
                            subfeature_name]
                        features[instance_index,
                                 feature_index] = subfeature_value
                    except KeyError:
                        logging.debug('Ignoring unknown subfeature: %s'
                                      % subfeature_name)

        features = features.tocsr()
        elapsed_seconds = time.time() - start_time
        logging.debug('Done featurizing in %0.2f seconds' % elapsed_seconds)
        if self.save_featurized:
            self.features = features
        return features

    @staticmethod
    def get_selected_features(feature_name_dictionary):
        selected_features = set()
        for feature_string in feature_name_dictionary.ids_to_names:
            feature_names = (
                FeatureExtractor.separate_conjoined_feature_names(
                    feature_string))
            # Split by conjoined features, and take all names as the
            # selected features. (Each feature name is of the form
            # name=value.)
            # print feature_string, 'Names:', feature_names
            feature_names = [name.split('=')[0] for name in feature_names]
            conjoined_feature_name = FeatureExtractor.conjoin_feature_names(
                feature_names)
            selected_features.add(conjoined_feature_name)
        return selected_features

    def _initialize_feature_extractors(self, selected_features):
        self.feature_extractors = []
        for feature_name in selected_features:
            extractor_names = FeatureExtractor.separate_conjoined_feature_names(
                feature_name)
            if len(extractor_names) > 1:
                # Grab the extractors in the order they were specified, so that
                # upon extraction the order of their conjoined features will
                # match.
                extractors = []
                try:
                    for name in extractor_names:
                        extractor = (e for e in self.all_feature_extractors
                                     if e.name == name).next()
                        extractors.append(extractor)
                except StopIteration:
                    raise FeaturizationError("Invalid conjoined feature name: %s"
                                             % feature_name)
                self.feature_extractors.append(
                    self.ConjoinedFeatureExtractor(feature_name, extractors))
            else:
                try:
                    # Find the correct extractor with a generator, which will
                    # stop searching once it's found.
                    extractor_generator = (
                        extractor for extractor in self.all_feature_extractors
                        if extractor.name == feature_name)
                    extractor = extractor_generator.next()
                    self.feature_extractors.append(extractor)
                except StopIteration:
                    raise FeaturizationError("Invalid feature name: %s"
                                             % feature_name)


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

    def extract_subfeature_names(self, instances):
        ''' Ignore `instances` and just use known values from initialization. '''
        return [FeatureExtractor._get_categorical_feature_name(
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

    def train(self, instances):
        self.training_results = self._trainer(instances)
        subfeature_extractors = self._feature_extractor_creator(
            self.training_results)
        self._subfeature_extractors = []
        sep = FLAGS.subfeature_sep
        for subfeature_name, subfeature_extractor in subfeature_extractors:
            full_subfeature_name = '%s%s%s' % (self.name, sep, subfeature_name)
            subfeature_extractor = FeatureExtractor(
                full_subfeature_name, subfeature_extractor, self.feature_type)
            self._subfeature_extractors.append(subfeature_extractor)

    def extract_subfeature_names(self, instances):
        assert self._subfeature_extractors is not None, (
            "Cannot retrieve subfeature names before training")
        subfeature_names = [e.extract_subfeature_names(instances)
                            for e in self._subfeature_extractors]
        return list(itertools.chain.from_iterable(subfeature_names))

    def extract(self, part):
        subfeature_values = [e.extract(part)
                             for e in self._subfeature_extractors]
        return merge_dicts(subfeature_values)


class SetValuedFeatureExtractor(FeatureExtractor):
    '''
    Class for extracting features where the same feature name can legally take
    on multiple discrete values simultaneously -- i.e., the feature is
    set-valued -- but where we want to represent that set as a collection of
    individual indicator features, rather than a single indicator for each
    possible set value.
    '''

    def __init__(self, name, extractor_fn):
        '''
        Unlike a traditional feature extractor, extractor_fn for a
        SetValuedFeatureExtractor should return a list (or tuple) of values.
        '''
        super(SetValuedFeatureExtractor, self).__init__(
            name, extractor_fn, self.FeatureTypes.Categorical)

    def extract_subfeature_names(self, instances):
        values_set = set()
        for part in instances:
            values_set.update(self._extractor_fn(part))
        return [FeatureExtractor._get_categorical_feature_name(self.name, value)
                for value in values_set]

    def extract(self, part):
        '''
        Returns a dictionary of subfeature name -> subfeature value. More
        complex feature extractor classes should override this function.
        '''
        feature_values = self._extractor_fn(part)
        feature_names = [self._get_categorical_feature_name(self.name, value)
                         for value in feature_values]
        return {feature_name: 1.0 for feature_name in feature_names}


class VectorValuedFeatureExtractor(FeatureExtractor):
    '''
    Class for extracting vector-valued features, where each vector is a fixed
    size and we want to record a feature for every position in the vector.
    '''
    def __init__(self, name, extractor_fn):
        super(VectorValuedFeatureExtractor, self).__init__(
            name, extractor_fn, self.FeatureTypes.Numerical)

    def extract_subfeature_names(self, instances):
        # NOTE: Assumes at least one part.
        vector_size = len(self._extractor_fn(instances[0]))
        return ['%s[%d]' % (self.name, i) for i in range(vector_size)]

    def extract(self, part):
        feature_values = {}
        for i, subfeature_value in enumerate(self._extractor_fn(part)):
            subfeature_name = '%s[%d]' % (self.name, i)
            feature_values[subfeature_name] = subfeature_value
        return feature_values
