""" Define standard machine-learned model framework for pipelines. """

from gflags import DEFINE_bool, DEFINE_string, FLAGS, DuplicateFlagError
import itertools
import logging
import numpy as np
import pycrfsuite
from scipy.sparse import lil_matrix, vstack
import time
from types import MethodType

from pipeline.featurization import FeatureExtractor, Featurizer, \
    FeaturizationError
from util import NameDictionary, listify
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
        train_result = self._train_model(instances)
        self._post_model_train(train_result)

    def test(self, instances):
        '''
        Returns an iterable of predicted outputs for the provided instances. If
        incremental output is desired, this should return a generator.
        '''
        raise NotImplementedError

    def save(self, filepath):
        raise NotImplementedError

    def load(self, filepath):
        logging.info("Loading model from %s...", filepath)
        load_result = self._load_model(filepath)
        logging.info("Done loading model.")
        self._post_model_load(load_result)

    def _load_model(self, filepath):
        raise NotImplementedError

    def _train_model(self, instances):
        raise NotImplementedError

    def _post_model_load(self, load_result):
        pass

    def _post_model_train(self, train_result):
        pass

    def reset(self):
        pass


class FeaturizedModel(Model):
    '''
    Subclasses' _load_model function should return a FeatureNameDictionary or a
    list of them, or else the _post_model_load hook must be overridden.
    '''

    def __init__(self, selected_features, model_path, save_featurized):
        self.save_featurized = save_featurized

        if model_path:
            self.load(model_path)
        else:
            if selected_features is None:
                raise FeaturizationError(
                    'Featurized model must be initialized with either selected'
                    ' features or a model path')

    # Subclasses should override this class-level variable to include actual
    # feature extractor objects.
    all_feature_extractors = []


class ClassifierModel(FeaturizedModel):
    def __init__(self, classifier, selected_features=None,
                 model_path=None, save_featurized=False):
        """
        Note that classifier must support the fit and predict methods in the
        style of scikit-learn.
        """
        self.classifier = classifier
        super(ClassifierModel, self).__init__(
            selected_features, model_path, save_featurized)
        if not model_path:
            self.featurizer = Featurizer(self.all_feature_extractors,
                                         selected_features,
                                         self.save_featurized)

    def _post_model_load(self, feature_name_dictionary):
        super(ClassifierModel, self)._post_model_load(feature_name_dictionary)
        self.featurizer = Featurizer(
            self.all_feature_extractors, feature_name_dictionary,
            self.save_featurized)

    def reset(self):
        self.featurizer.reset()

    def train(self, instances):
        self.reset() # Reset state in case we've been previously trained.
        logging.info("Registering features...")
        self.featurizer.register_features_from_instances(instances)
        logging.info('Done registering features.')

        features = self.featurize(instances)
        labels = self._get_gold_labels(instances)
        logging.info('Fitting classifier...')
        self.classifier.fit(features, labels)
        logging.info('Done fitting classifier.')

    def test(self, instances):
        features = self.featurizer.featurize(instances)
        gold_labels = self._get_gold_labels(instances)
        if self.save_featurized:
            self.gold_labels = gold_labels
        labels = self.classifier.predict(features)
        # logging.debug('%d data points' % len(gold_labels))
        # logging.debug('Raw classifier performance:')
        # logging.debug('\n' + str(diff_binary_vectors(labels, gold_labels)))
        return labels

    def _get_gold_labels(self, instances):
        raise NotImplementedError


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

    def __init__(self, part_type, model_file_path,
                 selected_features, training_algorithm, training_params,
                 save_model_info=False):
        super(CRFModel, self).__init__(part_type)
        self.model_file_path = model_file_path
        self.feature_extractors = [
            extractor for extractor in self.all_feature_extractors
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
