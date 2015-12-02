""" Define standard machine-learned model framework for pipelines. """

from gflags import DEFINE_bool, DEFINE_string, FLAGS, DuplicateFlagError
import itertools
import logging
import numpy as np
import pycrfsuite
from scipy.sparse import lil_matrix, vstack
import time
from types import MethodType

from pipeline.feature_extractors import FeatureExtractor, FeatureExtractionError
from util import NameDictionary
# from util.metrics import diff_binary_vectors

try:
    DEFINE_bool(
        'rebalance_stochastically', False,
        'Rebalance classes by stochastically choosing samples to replicate')
    DEFINE_bool('pycrfsuite_verbose', False,
                'Verbose logging output from python-crfsuite trainer')
except DuplicateFlagError as e:
    logging.warn('Ignoring redefinition of flag %s' % e.flagname)


class Model(object):
    def train(self, instances):
        raise NotImplementedError

    def test(self, instances):
        raise NotImplementedError

    def save(self, filepath):
        raise NotImplementedError

    def load(self, filepath):
        raise NotImplementedError

    def reset(self):
        pass


class FeaturizationError(Exception):
    pass


class FeaturizedModel(Model):
    class ConjoinedFeatureExtractor(FeatureExtractor):
        '''
        A FeaturizedModel allows combining features. This class provides the
        necessary functionality of combining the FeatureExtractors' outputs.
        '''

        def __init__(self, name, extractors):
            self.name = name
            self.feature_type = self.FeatureTypes.Categorical
            for extractor in extractors:
                if extractor.feature_type != self.FeatureTypes.Categorical:
                    raise FeatureExtractionError(
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

    def __init__(self, feature_extractors, selected_features=None,
                 model_path=None, save_featurized=False):
        """
        feature_extractors is a list of
            `pipeline.feature_extractors.FeatureExtractor` objects.
        selected_features is a list of names of features to extract. Names may
            be combinations of feature names, separated by
            FLAGS.conjoined_feature_sep. (This character should be escaped in
            the names of any conjoined features containing it, or the conjoined
            features may not work properly.)
        save_featurized indicates whether to store features and labels
            properties after featurization. Useful for debugging/development.
        """
        self.feature_name_dictionary = NameDictionary()
        self.all_feature_extractors = feature_extractors
        self.feature_training_data = None
        self.save_featurized = save_featurized

        if model_path is not None:
            selected_features = self.load(model_path)
            if selected_features is not None:
                logging.warn("Selected features overridden by loaded model")
        elif selected_features is None:
            raise FeaturizationError(
                "FeaturizedModel must be initialized with either selected"
                " features or a model path to load")
        else: # if not loading a model, initialize feature extractors directly.
            self._initialize_feature_extractors(selected_features)

    def load(self, filepath):
        self.reset()
        logging.info("Loading model from %s...", filepath)
        selected_features = self._featurized_load(filepath)
        logging.info("Done loading model.")
        self._initialize_feature_extractors(selected_features)

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

    def _featurized_load(self, filepath):
        '''
        Does the actual work of loading the model from a file. Populates
        self.feature_name_dictionary and any model parameters. Returns the list
        of features that were selected in the saved model. Must be overridden.
        '''
        raise NotImplementedError

    def reset(self):
        self.feature_name_dictionary.clear()
        self.feature_training_data = None

    def train(self, instances):
        self.reset() # Reset state in case we've been previously trained.
        logging.info("Registering features...")
        self._register_features(instances)
        logging.info('Done registering features.')
        self._featurized_train(instances)

    def _register_features(self, instances):
        # Build feature name dictionary. (Unfortunately, this means we
        # featurize many things twice, but I can't think of a cleverer way to
        # discover the possible values of a feature.)
        for extractor in self.feature_extractors:
            self.feature_training_data = extractor.train(instances)
            subfeature_names = extractor.extract_subfeature_names(instances)
            for subfeature_name in subfeature_names:
                self.feature_name_dictionary.insert(subfeature_name)

    def test(self, instances):
        # TODO: eliminate the extra level of indirection here
        assert self.feature_name_dictionary, (
            "Feature name dictionary must be populated either by training or by"
            " loading a model")
        return self._featurized_test(instances)

    def _featurized_train(self, instances):
        '''
        The exact training procedure depends on the type of model. This method
        must be overridden by subclasses.
        '''
        raise NotImplementedError

    def _featurized_test(self, instances):
        '''
        The exact testing procedure depends on the type of model. This method
        must be overridden by subclasses.
        '''
        raise NotImplementedError

    def _featurize(self, instances):
        '''
        The process of generating a feature matrix from a bunch of instances
        is common to all models, even though they may do different things with
        the resulting matrix.
        '''
        logging.debug('Featurizing...')
        start_time = time.time()

        features = lil_matrix(
            (len(instances), len(self.feature_name_dictionary)),
            dtype=np.float32) # TODO: Make this configurable?

        for extractor in self.feature_extractors:
            feature_values_by_instance = extractor.extract_all(instances)
            for instance_index, instance_subfeature_values in enumerate(
                feature_values_by_instance):
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


class ClassifierModel(FeaturizedModel):
    def __init__(self, classifier, *args, **kwargs):
        """
        Note that classifier must support the fit and predict methods in the
        style of scikit-learn.
        """
        super(ClassifierModel, self).__init__(*args, **kwargs)
        self.classifier = classifier

    # TODO: fix training to get labels from somewhere
    def _featurize(self, instances):
        features = super(ClassifierModel, self)._featurize(instances)
        labels = np.fromiter((instance.label for instance in instances), int,
                             len(instances))
        if self.save_featurized:
            self.labels = labels
        return features, labels

    def _featurized_train(self, instances):
        features, labels = self._featurize(instances)
        logging.info('Fitting classifier...')
        self.classifier.fit(features, labels)
        logging.info('Done fitting classifier.')

    def _featurized_test(self, instances):
        features, gold_labels = self._featurize(instances)
        if self.save_featurized:
            self.gold_labels = gold_labels
        labels = self.classifier.predict(features)
        # logging.debug('%d data points' % len(gold_labels))
        # logging.debug('Raw classifier performance:')
        # logging.debug('\n' + str(diff_binary_vectors(labels, gold_labels)))
        return labels


class ClassBalancingClassifierWrapper(object):
    def __init__(self, classifier, ratio=float('inf')):
        self.classifier = classifier
        self.ratio = ratio

    @staticmethod
    def rebalance(data, labels, ratio=float('inf')):
        """
        data is a sparse matrix; labels is array-like.
        ratio indicates the maximum ratio of its current count to which any
        class is allowed to increase.
        """
        if ratio <= 1.0: # No increase
            return data, labels

        # Based on http://stackoverflow.com/a/23392678/4044809
        label_set, label_indices, label_counts = np.unique(
            labels, return_inverse=True, return_counts=True)

        max_count = label_counts.max()
        counts_to_add = [
            # -1 adjusts for the current_count we already have
            int(min(max_count - current_count, (ratio - 1) * current_count))
            for current_count in label_counts]
        rows_to_add = np.sum(counts_to_add)
        if rows_to_add == 0:
            # This is essential not just for efficiency, but because scipy's
            # vstack apparently doesn't know how to vstack a 0-row matrix.
            return data, labels

        # Use lil_matrix to support slicing.
        rebalanced_data = lil_matrix((rows_to_add, data.shape[1]),
                                     dtype=data.dtype)
        rebalanced_labels = np.empty((rows_to_add,), labels.dtype)

        slices = np.concatenate(([0], np.cumsum(counts_to_add)))
        for j in xrange(len(label_set)):
            label_row_indices = np.where(label_indices == j)[0]
            if FLAGS.rebalance_stochastically:
                indices = np.random.choice(label_row_indices,
                                           counts_to_add[j])
            else:
                full_repetitions = (
                    counts_to_add[j] / label_row_indices.shape[0])
                logging.debug("Increasing count for label %d by %d+"
                              % (j, full_repetitions))
                indices = np.tile(label_row_indices, (full_repetitions,))
                still_needed = counts_to_add[j] - indices.shape[0]
                if still_needed:
                    indices = np.concatenate(
                        (indices, label_row_indices[:still_needed]))

            # Only bother to actually rebalance if there are changes to be made
            if indices.shape[0]:
                rebalanced_data[slices[j]:slices[j + 1]] = data[indices]
                rebalanced_labels[slices[j]:slices[j + 1]] = labels[indices]
        rebalanced_data = vstack((data, rebalanced_data))
        rebalanced_labels = np.concatenate((labels, rebalanced_labels))
        return (rebalanced_data, rebalanced_labels)

    def fit(self, data, labels):
        rebalanced_data, rebalanced_labels = self.rebalance(
            data, labels, self.ratio)
        return self.classifier.fit(rebalanced_data, rebalanced_labels)

    def predict(self, data):
        return self.classifier.predict(data)


# TODO: redo this to match new document/instance/parts structure
class CRFModel(Model):
    # Theoretically, this class ought to be a type of FeaturizedModel. But that
    # class assumes we're doing all the feature management in class, and in
    # practice we're offloading CRF feature management to CRFSuite.

    class CRFTrainingError(Exception):
        pass

    class ObservationWithContext(object):
        '''
        In a CRF model, the feature extractors will have to operate not just on
        a single part (which may have many observations), but on each
        observation, *with* the context of the surrounding observations. This
        class encapsulates the data a CRF feature extractor may need.
        '''
        def __init__(self, observation, sequence, index, part):
            self.observation = observation
            self.sequence = sequence
            self.index = index
            self.part = part

    def __init__(self, part_type, model_file_path, feature_extractors,
                 selected_features, training_algorithm, training_params,
                 save_model_info=False):
        super(CRFModel, self).__init__(part_type)
        self.model_file_path = model_file_path
        self.feature_extractors = [extractor for extractor in feature_extractors
                                   if extractor.name in selected_features]
        self.training_algorithm = training_algorithm
        self.training_params = training_params
        self.save_model_info = save_model_info
        self.model_info = None

    def _sequences_for_instance(self, part, is_train):
        '''
        Returns the observation and label sequences for the part. Should return
        None for labels at test time.
        '''
        raise NotImplementedError

    def _label_part(self, part, crf_labels):
        '''
        Applies the labels to the part as appropriate. Must be overridden.
        (In many cases with CRFs, this will involve interpreting the labels
        as spans.)
        '''
        raise NotImplementedError

    def __featurize_observation_sequence(self, observation_sequence, part):
        observation_features = []
        for i, observation in enumerate(observation_sequence):
            feature_values = {}
            for feature_extractor in self.feature_extractors:
                extractor_arg = self.ObservationWithContext(
                    observation, observation_sequence, i, part)
                feature_values.update(feature_extractor.extract(extractor_arg))
            observation_features.append(feature_values)
        return observation_features

    @staticmethod
    def __handle_training_error(trainer, log):
        raise CRFModel.CRFTrainingError('CRF training failed: %s' % log)

    def train(self, instances):
        trainer = pycrfsuite.Trainer(verbose=FLAGS.pycrfsuite_verbose)
        trainer.select(self.training_algorithm)
        trainer.set_params(self.training_params)
        error_handler = MethodType(self.__handle_training_error, trainer)
        trainer.on_prepare_error = error_handler

        for instances in instances:
            observation_sequence, labels = self._sequences_for_instance(
                instances, True)
            observation_features = self.__featurize_observation_sequence(
                observation_sequence, instances)
            trainer.append(observation_features, labels)

        start_time = time.time()
        logging.info("Training CRF model...")
        trainer.train(self.model_file_path)
        elapsed_seconds = time.time() - start_time
        logging.info('CRF model saved to %s (training took %0.2f seconds)'
                     % (self.model_file_path, elapsed_seconds))
        if self.save_model_info:
            tagger = pycrfsuite.Tagger()
            tagger.open(self.model_file_path)
            self.model_info = tagger.info()
            tagger.close()

    def test(self, instances):
        tagger = pycrfsuite.Tagger()
        tagger.open(self.model_file_path)
        instance_labels = []
        for instance in instances:
            observation_sequence, _ = self._sequences_for_instance(instance, False)
            observation_features = self.__featurize_observation_sequence(
                observation_sequence, instance)
            crf_labels = tagger.tag(observation_features)
            instance_labels.append(crf_labels)
        tagger.close()
        return instance_labels
