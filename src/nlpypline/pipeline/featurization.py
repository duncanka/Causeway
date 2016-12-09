from collections import Counter, defaultdict
from copy import copy
from cPickle import PicklingError
from gflags import DEFINE_string, FLAGS, DuplicateFlagError
from itertools import chain, product
import logging
import numpy as np
from scipy.sparse import lil_matrix
import time

from nlpypline.util import (Enum, NameDictionary, get_object_by_fqname,
                            merge_dicts)
from pandas.core.algorithms import isin

try:
    DEFINE_string('conjoined_feature_sep', ':',
                  'Separator character to use between conjoined feature names.'
                  ' This character can still be used in conjoined feature names'
                  ' by doubling it (e.g., "f1=:::f2=x").')
except DuplicateFlagError as e:
    logging.warn('Ignoring flag redefinitions; assuming module reload')


class FeaturizationError(Exception):
    pass


class FeatureExtractor(object):
    FeatureTypes = Enum(['Categorical', 'Numerical', 'Binary'])
    '''
    Whether extract() can return features not registered by
    extract_subfeature_names when run on the same set of instances. Should be
    overridden in extractor classes where this is true.
    '''
    _EXTRACT_PRODUCES_VALUES_TO_IGNORE = False

    def __init__(self, name, extractor_fn, feature_type=None):
        if feature_type is None:
            feature_type = self.FeatureTypes.Categorical
        self.name = name
        self.feature_type = feature_type
        self._extractor_fn = extractor_fn

    def extract_subfeature_names(self, instances):
        if self.feature_type == self.FeatureTypes.Categorical:
            values_set = set(self._extractor_fn(part) for part in instances)
            return [self._get_categorical_feature_name(self.name, value)
                    for value in values_set]
        else: # feature_type == Numerical or feature_type == Binary
            return [self.name]

    def extract(self, part):
        '''
        Returns a dictionary of subfeature name -> subfeature value. More
        complex feature extractor classes should override this function.
        '''
        feature_value = self._extractor_fn(part)
        if self.feature_type == self.FeatureTypes.Categorical:
            feature_name = self._get_categorical_feature_name(
                self.name, feature_value)
            return {feature_name: 1.0}
        else: # feature_type == Numerical or feature_type == Binary
            return {self.name: feature_value}

    def extract_all(self, parts):
        return [self.extract(part) for part in parts]

    @staticmethod
    def _get_categorical_feature_name(base_name, value):
        return '%s=%s' % (base_name, value)

    def __repr__(self):
        return '<Feature extractor: %s>' % self.name


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
        ''' Ignore `instances` and just use known values. '''
        return [self._get_categorical_feature_name(self.name, value)
                for value in self.feature_values]


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
        return [self._get_categorical_feature_name(self.name, value)
                for value in values_set]

    def extract(self, part):
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
        # NOTE: Assumes at least one instance.
        vector_size = len(self._extractor_fn(instances[0]))
        return ['%s[%d]' % (self.name, i) for i in range(vector_size)]

    def extract(self, part):
        feature_values = {}
        for i, subfeature_value in enumerate(self._extractor_fn(part)):
            subfeature_name = '%s[%d]' % (self.name, i)
            feature_values[subfeature_name] = subfeature_value
        return feature_values


class NestedFeatureExtractor(FeatureExtractor):
    def __init__(self, name, subextractors):
        self.name = name
        self.subextractors = subextractors
        self.feature_type = subextractors[0].feature_type
        if not all(e.feature_type == self.feature_type for e in subextractors):
            self.feature_type = -1 # undefined...

    def extract_subfeature_names(self, instances):
        names_by_subextractor = [
            ['_'.join([self.name, subextractor_subfeature_name])
             for subextractor_subfeature_name
             in subextractor.extract_subfeature_names(instances)]
            for subextractor in self.subextractors]
        return list(chain.from_iterable(names_by_subextractor))

    def extract(self, part):
        features_by_subextractor = [
            {'_'.join([self.name, feature_name]): feature_value
             for feature_name, feature_value
             in subextractor.extract(part).iteritems()}
            for subextractor in self.subextractors]
        return merge_dicts(features_by_subextractor)


class MultiNumericalFeatureExtractor(FeatureExtractor):
    '''
    Feature extractor for extracting a feature with multiple named numerical
    components. Expects extractor_fn to return the feature-name-to-numerical-
    value dictionary, though feature names will be updated with the extractor
    name.
    '''
    def __init__(self, name, extractor_fn):
        FeatureExtractor.__init__(self, name, extractor_fn,
                                  self.FeatureTypes.Numerical)

    def extract_subfeature_names(self, instances):
        names_set = set()
        for part in instances:
            names_set.update([self._complete_feature_name(key)
                              for key in self._extractor_fn(part).keys()])
        return names_set

    def extract(self, part):
        ''' Directly returns the dictionary of feature names/values. '''
        return {self._complete_feature_name(key): val
                for key, val in self._extractor_fn(part).iteritems()}

    def _complete_feature_name(self, feature_name):
        return '='.join([self.name, feature_name])


class _MetaFeatureExtractor(FeatureExtractor):
    """
    Base type for feature extractors that transform other feature extractors.
    """
    def __init__(self, base_extractor):
        self.feature_type = base_extractor.feature_type
        self.name = base_extractor.name
        self.base_extractor = base_extractor

    def extract(self, part):
        return self.base_extractor.extract(part)


class ThresholdedFeatureExtractor(_MetaFeatureExtractor):
    """
    Takes another FeatureExtractor and turns it into one that only extracts
    features that appear at least a given number of times in the training data.
    This class doesn't respect the base extractor's extract_subfeature_names.
    """

    _EXTRACT_PRODUCES_VALUES_TO_IGNORE = True

    def __init__(self, base_extractor, threshold):
        """
        If threshold is a string, it will be treated as a flag name to be read
        from FLAGS the first time it is needed.
        """
        super(ThresholdedFeatureExtractor, self).__init__(base_extractor)
        self.threshold = threshold

    def extract_subfeature_names(self, instances):
        if isinstance(self.threshold, basestring):
            self.threshold = getattr(FLAGS, self.threshold)

        names_to_counts = Counter()
        for instance in instances:
            for k in self.base_extractor.extract(instance).keys():
                names_to_counts[k] += 1
        return [k for k, v in names_to_counts.iteritems()
                if v >= self.threshold]

    # Don't worry about extracting things that didn't appear often enough.
    # Features not meeting the threshold will just be ignored as unknown
    # features.


class BinnedFeatureExtractor(_MetaFeatureExtractor):
    """
    Takes a numerical FeatureExtractor and turns it into a categorical feature
    extractor with bins for the values.
    """
    DEFAULT_BIN_FN = lambda x: [int(x), int(x) + 1]

    def __init__(self, base_extractor, bin_fn=DEFAULT_BIN_FN):
        if base_extractor.feature_type != self.FeatureTypes.Numerical:
            raise FeaturizationError("Can't bin non-numerical feature %s"
                                     % base_extractor.name)
        base_extractor = copy(base_extractor) # avoid overwriting extractor fn
        super(BinnedFeatureExtractor, self).__init__(base_extractor)
        self.feature_type = self.FeatureTypes.Categorical
        self.bin_fn = bin_fn

    def extract(self, part):
        return {self._get_binned_feature_name(k, self.bin_fn(v)): 1
                for k, v in self.base_extractor.extract(part).iteritems()}

    def extract_subfeature_names(self, instances):
        # We need to know the actual values to know what the bins will be.
        names = set()
        if self.base_extractor._EXTRACT_PRODUCES_VALUES_TO_IGNORE:
            # Well, great. Now we have to run extraction twice, to get the bin
            # values and to filter to extracted features that we actually care
            # about.
            valid_names = set(self.base_extractor.extract_subfeature_names(
                instances))
            for part in instances:
                for k, v in self.base_extractor.extract(part).iteritems():
                    if k in valid_names:
                        names.add(
                            self._get_binned_feature_name(k, self.bin_fn(v)))
        else:
            for part in instances:
                names.update(self.extract(part).keys())

        return names

    @staticmethod
    def _get_binned_feature_name(base_name, bin_id):
        return u'%s<%s' % (base_name, bin_id)

    @staticmethod
    def bin_all_numeric(extractors, bin_fn=DEFAULT_BIN_FN):
        return [(e if e.feature_type != FeatureExtractor.FeatureTypes.Numerical
                 else BinnedFeatureExtractor(e, bin_fn))
                for e in extractors]


class Featurizer(object):
    '''
    Encapsulates and manages a set of FeatureExtractors, stores the feature name
    dictionary, and actually runs extractors to produce a feature matrix.
    '''
    @staticmethod
    def check_feature_name(feature_name, extractor_names, conjoined_sep):
        return (feature_name == 'all' or all(
                name in extractor_names for name in
                Featurizer.separate_conjoined_feature_names(feature_name,
                                                            conjoined_sep)))

    @staticmethod
    def selected_features_for_featurizer(selected_features,
                                         extractors_for_featurizer):
        """
        From a list of selected feature names, return only the ones relevant to
        a particular Featurizer's extractor list. (Only the extractor list is
        required, not the Featurizer.)
        """
        conjoined_sep = FLAGS.conjoined_feature_sep
        extractor_names = [e.name for e in extractors_for_featurizer]
        selected_features = [
            feature for feature in selected_features
            if Featurizer.check_feature_name(feature, extractor_names,
                                             conjoined_sep)]
        return selected_features

    @staticmethod
    def check_selected_features_list(selected_features, all_extractors):
        """
        Check selected features against a complete list of all available
        features. Useful in cases where individual Featurizers don't have
        access to all FeatureExtractors to check the list of names.
        """
        conjoined_sep = FLAGS.conjoined_feature_sep
        extractor_names = [e.name for e in all_extractors]
        for feature in selected_features:
            if not Featurizer.check_feature_name(feature, extractor_names,
                                                 conjoined_sep):
                raise FeaturizationError("Invalid feature name: %s" % feature)

    @staticmethod
    def escape_conjoined_name(feature_name, sep):
        return feature_name.replace(sep, sep * 2)

    @staticmethod
    def separate_conjoined_feature_names(conjoined_names, sep):
        ''' Returns unescaped split feature names. '''
        double_sep = sep * 2
        conjoined_names = conjoined_names.replace(double_sep, '\0')
        return [name.replace('\0', sep)
                for name in conjoined_names.split(sep)]

    @staticmethod
    def conjoin_feature_names(feature_names, sep):
        return sep.join(feature_names)

    class ConjoinedFeatureExtractor(FeatureExtractor):
        '''
        A Featurizer allows combining features. This class provides the
        necessary functionality of combining the FeatureExtractors' outputs.
        '''

        def __init__(self, name, extractors):
            self.name = name
            self.feature_type = self.FeatureTypes.Categorical
            for extractor in extractors:
                if extractor.feature_type == self.FeatureTypes.Numerical:
                    raise FeaturizationError(
                        "Numerical features cannot be conjoined (attempted to"
                        " conjoin %s)" % [e.name for e in extractors])
            self._extractors = extractors
            self.sep = FLAGS.conjoined_feature_sep

        # NOTE: this overrides the normal extract_subfeature_names with a
        # DIFFERENT SIGNATURE. That's OK, because we only ever use this class
        # internally.
        def extract_subfeature_names(self, instances, names_by_extractor):
            subfeature_name_components = [
                names_by_extractor[extractor] for extractor in self._extractors]
            return [Featurizer.conjoin_feature_names(cmpts, self.sep) for
                    cmpts in product(*subfeature_name_components)]

        # NOTE: likewise for this one -- it overrides the default signature.
        def extract(self, instance, featurized_cache=None):
            # Sub-extractors may produce entire dictionaries of feature values
            # (e.g., for set-valued feature extractors). We need to produce one
            # conjoined feature for every element of the Cartesian product of
            # these dictionaries.
            # Note that the extractor results come out in the same order as the
            # features were initially specified, so we can safely construct the
            # conjoined name by just joining these subfeature names.
            if featurized_cache:
                extractor_results = [
                    featurized_cache[extractor]
                    for extractor in self._extractors]
                # Separator chars are already escaped.
            else:
                # Categorical and binary features are all binary values, so if
                # they're present in the results dict, they're firing.
                extractor_results = [extractor.extract(instance).keys()
                                     for extractor in self._extractors]
                # Separator characters must be escaped for conjoined names.
                extractor_results = [
                    [Featurizer.escape_conjoined_name(name, self.sep)
                     for name in extractor_result]
                    for extractor_result in extractor_results]
            cartesian_product = product(*extractor_results)
            return {self.sep.join(subfeature_names): 1.0
                    for subfeature_names in cartesian_product}


    def __init__(self, feature_extractors, selected_features_or_name_dict,
                 instance_filter=None, save_featurized=False,
                 default_to_matrix=True):
        """
        feature_extractors is a list of
            `pipeline.featurization.FeatureExtractor` objects, or the fully-
            qualified name of such a list. It MUST be the latter if the
            Featurizer is to be pickled.
        selected_features_or_name_dict must be one of the following:
            - a list of names of features to extract. Names may be combinations
              of feature names, separated by FLAGS.conjoined_feature_sep. (This
              character should be escaped in the names of any conjoined features
              containing it, or the conjoined features may not work properly.)
              The special name 'all' activates all non-conjoined features.
            - a NameDictionary that should be used for this Featurizer. This
              implicitly encodes the selected features.
        instance_filter is a filter function that takes an instance and returns
            True iff it should be featurized. Instances that are filtered out
            will be featurized as all zeros.
        save_featurized indicates whether to store features and labels
            properties after featurization. Useful for debugging/development.
        default_to_matrix indicates whether featurize() should by default return
            a matrix, rather than a raw dictionary of feature names and values.
        """
        if isinstance(feature_extractors, basestring):
            self.feature_extractors_fqname = feature_extractors
            feature_extractors = get_object_by_fqname(feature_extractors)
        else:
            self.feature_extractors_fqname = None

        self.all_feature_extractors = feature_extractors
        self.save_featurized = save_featurized
        self.default_to_matrix = default_to_matrix

        self._instance_filter = instance_filter

        if isinstance(selected_features_or_name_dict, NameDictionary):
            self.feature_name_dictionary = selected_features_or_name_dict
            selected_features = self.get_selected_features(
                selected_features_or_name_dict)
        else:
            self.feature_name_dictionary = NameDictionary()
            selected_features = selected_features_or_name_dict
        self._initialize_feature_extractors(selected_features)

        self.featurized = None

    def reset(self):
        self.feature_name_dictionary.clear()

    def register_feature_names(self, feature_names):
        for feature_name in feature_names:
            self.feature_name_dictionary.insert(feature_name)

    def register_features_from_instances(self, instances):
        # Build feature name dictionary. (Unfortunately, this means we
        # featurize many things twice, but I can't think of a cleverer way to
        # discover the possible values of a feature.)
        sep = FLAGS.conjoined_feature_sep
        names_by_extractor = {}
        for extractor in self._selected_base_extractors:
            subfeature_names = extractor.extract_subfeature_names(instances)
            subfeature_names = [
                Featurizer.escape_conjoined_name(subfeature_name, sep)
                for subfeature_name in subfeature_names]
            self.register_feature_names(subfeature_names)
            names_by_extractor[extractor] = subfeature_names

        for extractor in self._unselected_base_extractors:
            subfeature_names = extractor.extract_subfeature_names(instances)
            subfeature_names = [
                Featurizer.escape_conjoined_name(subfeature_name, sep)
                for subfeature_name in subfeature_names]
            names_by_extractor[extractor] = subfeature_names

        for extractor in self._conjoined_extractors:
            subfeature_names = extractor.extract_subfeature_names(
                instances, names_by_extractor)
            self.register_feature_names(subfeature_names)

    def __record_subfeatures(self, instance_subfeature_values, instance_index,
                             features):
        for subfeature_name, subfeature_value in (
            instance_subfeature_values.items()):
            if subfeature_value == 0:
                continue # Don't bother setting 0's in a sparse matrix.
            try:
                feature_index = self.feature_name_dictionary[subfeature_name]
                features[instance_index, feature_index] = subfeature_value
            except KeyError:
                pass
                # logging.debug('Ignoring unknown subfeature: %s'
                #              % subfeature_name)

    def featurize(self, instances, to_matrix=None):
        if to_matrix is None:
            to_matrix = self.default_to_matrix

        logging.debug('Featurizing...')
        start_time = time.time()

        if to_matrix:
            features = lil_matrix(
                (len(instances), len(self.feature_name_dictionary)),
                dtype=np.float32) # TODO: Make this configurable?
        else:
            features = [{} for _ in instances]

        fresh_featurized_cache = dict.fromkeys(
            self._unselected_base_extractors.union(
                self._selected_base_extractors), None)
        sep = FLAGS.conjoined_feature_sep

        for instance_index, instance in enumerate(instances):
            if self._instance_filter and not self._instance_filter(instance):
                continue

            # Optimization: just copy the cache fresh each time
            featurized_cache = copy(fresh_featurized_cache)

            for extractor in self._unselected_base_extractors:
                instance_subfeature_values = extractor.extract(instance)
                escaped = [self.escape_conjoined_name(name, sep)
                           for name in instance_subfeature_values.keys()]
                featurized_cache[extractor] = escaped

            for extractor in self._selected_base_extractors:
                instance_subfeature_values = extractor.extract(instance)
                escaped = [self.escape_conjoined_name(name, sep)
                           for name in instance_subfeature_values.keys()]
                featurized_cache[extractor] = escaped
                if to_matrix:
                    self.__record_subfeatures(instance_subfeature_values,
                                              instance_index, features)
                else:
                    features[instance_index].update(instance_subfeature_values)

            for extractor in self._conjoined_extractors:
                instance_subfeature_values = extractor.extract(instance,
                                                               featurized_cache)
                if to_matrix:
                    self.__record_subfeatures(instance_subfeature_values,
                                              instance_index, features)
                else:
                    features[instance_index].update(instance_subfeature_values)

        if to_matrix:
            features = features.tocsr()
        elapsed_seconds = time.time() - start_time
        logging.debug('Done featurizing in %0.2f seconds' % elapsed_seconds)
        if self.save_featurized:
            self.featurized = features
        return features

    @staticmethod
    def get_selected_features(feature_name_dictionary):
        selected_features = set()
        sep = FLAGS.conjoined_feature_sep
        for feature_string in feature_name_dictionary.ids_to_names:
            feature_names = (
                Featurizer.separate_conjoined_feature_names(
                    feature_string, sep))
            # Split by conjoined features, and take all names as the
            # selected features. (Each feature name is of the form
            # name=value.)
            # print feature_string, 'Names:', feature_names
            feature_names = [name.split('=')[0] for name in feature_names]
            conjoined_feature_name = Featurizer.conjoin_feature_names(
                feature_names, sep)
            selected_features.add(conjoined_feature_name)
        return selected_features

    def __get_extractor_by_name(self, name):
        for extractor in self.all_feature_extractors:
            if extractor.name == name:
                return extractor
        raise KeyError

    def _initialize_feature_extractors(self, selected_features):
        self._unselected_base_extractors = set()
        self._selected_base_extractors = set()
        # TODO: should we make things slightly more efficient by not caching
        # base features that aren't part of some conjoined feature?
        self._conjoined_extractors = set()
        sep = FLAGS.conjoined_feature_sep

        if 'all' in selected_features:
            conjoined_names = [
                name for name in selected_features
                if len(self.separate_conjoined_feature_names(name, sep)) > 1]
            selected_features = ([e.name for e in self.all_feature_extractors]
                                 ) + conjoined_names

        for feature_name in selected_features:
            extractor_names = self.separate_conjoined_feature_names(
                feature_name, sep)
            if len(extractor_names) > 1:
                # Grab the extractors in the order they were specified, so that
                # upon extraction the order of their conjoined features will
                # match.
                try:
                    extractors = [self.__get_extractor_by_name(name)
                                  for name in extractor_names]
                except KeyError:
                    raise FeaturizationError(
                        "Invalid conjoined feature name: %s" % feature_name)

                conjoined = self.ConjoinedFeatureExtractor(feature_name,
                                                           extractors)
                self._conjoined_extractors.add(conjoined)

                for extractor in extractors:
                    if extractor.name not in selected_features:
                        self._unselected_base_extractors.add(extractor)

            else:
                try:
                    extractor = self.__get_extractor_by_name(feature_name)
                    self._selected_base_extractors.add(extractor)
                except KeyError:
                    raise FeaturizationError("Invalid feature name: %s"
                                             % feature_name)

    # Pickling functions

    def __getstate__(self):
        if not self.feature_extractors_fqname:
            raise PicklingError("Can't pickle a Featurizer without a fully-"
                                "qualified name from which to restore"
                                " extractors")

        state = self.__dict__.copy()
        for attr_name in [
            'featurized', 'all_feature_extractors', '_selected_base_extractors',
            '_unselected_base_extractors', '_conjoined_extractors']:
            del state[attr_name]
        state['selected_features'] = self.get_selected_features(
            self.feature_name_dictionary)
        return state

    def __setstate__(self, state):
        selected_features = state.pop('selected_features')
        self.__dict__.update(state)
        self.featurized = None
        self.all_feature_extractors = get_object_by_fqname(
            self.feature_extractors_fqname)
        self._initialize_feature_extractors(selected_features)

    # Support function, useful for debugging featurized results.
    def matrow2dict(self, features, row_index):
        row = features[row_index, :]
        return {self.feature_name_dictionary[int(i)]: row[0, i]
                for i in row.nonzero()[1]}


class DictOnlyFeaturizer(Featurizer):
    def __init__(self, feature_extractors, selected_features,
                 save_featurized=False, instance_filter=None):
        self.all_feature_extractors = feature_extractors
        self.save_featurized = save_featurized
        self._instance_filter = instance_filter
        self._initialize_feature_extractors(selected_features)

    def register_features_from_instances(self, instances):
        pass

    def reset(self):
        pass

    def featurize(self, instances, to_matrix=None):
        if to_matrix:
            raise FeaturizationError("Cannot featurize to matrix with %s" %
                                     self.__class__.__name__)
        return Featurizer.featurize(self, instances, False)
