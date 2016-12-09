""" Define standard machine-learned model framework for pipelines. """

import cPickle
from gflags import DEFINE_bool, FLAGS, DuplicateFlagError
import itertools
import logging
import numpy as np
from scipy.sparse import lil_matrix, vstack
from sklearn.base import BaseEstimator

from nlpypline.pipeline.featurization import (FeatureExtractor, Featurizer,
                                              FeaturizationError)
from nlpypline.util import NameDictionary, listify
# from nlpypline.util.metrics import diff_binary_vectors

try:
    DEFINE_bool(
        'rebalance_stochastically', False,
        'Rebalance classes by stochastically choosing samples to replicate')
except DuplicateFlagError as e:
    logging.warn('Ignoring flag redefinitions; assuming module reload')


class Model(object):
    def __init__(self, *args, **kwargs):
        if args or kwargs:
            logging.debug("Extra model arguments: args=%s, kwargs=%s",
                          args, kwargs)

    def train(self, instances):
        self.reset() # Reset state in case we've been previously trained.
        self._train_model(instances)
        self._post_model_train()

    # TODO: refactor so that overriding behavior is similar between train and
    # test (i.e., _test_model should be the thing to override)
    def test(self, instances):
        '''
        Returns an iterable of predicted outputs for the provided instances. If
        incremental output is desired, this should return a generator.
        '''
        raise NotImplementedError

    def save(self, filepath):
        # Default save implementation is to pickle the whole model.
        with open(filepath, 'wb') as pickle_file:
            cPickle.dump(self, pickle_file)

    def load(self, filepath):
        logging.debug("Loading model from %s...", filepath)
        self._load_model(filepath)
        logging.debug("Done loading model.")
        self._post_model_load()

    def _load_model(self, filepath):
        with open(filepath, 'r') as pickle_file:
            unpickled = cPickle.load(pickle_file)
            if type(unpickled) != type(self):
                raise cPickle.UnpicklingError(
                    'Attempted to load %s model from pickle of %s object'
                    % (type(self).__name__, type(unpickled).__name__))
            self.__dict__.update(unpickled.__dict__)

    def _train_model(self, instances):
        pass

    def _post_model_load(self):
        pass

    def _post_model_train(self):
        pass

    def reset(self):
        pass


class FeaturizedModelBase(Model):
    def __init__(self, selected_features_lists, model_path, save_featurized,
                 *args, **kwargs):
        super(FeaturizedModelBase, self).__init__(
            model_path=model_path, save_featurized=save_featurized,
            *args, **kwargs)
        self.save_featurized = save_featurized

        if model_path:
            self.load(model_path)
        else: # Featurizers won't be set up by loading
            if selected_features_lists is None:
                raise FeaturizationError(
                    'Featurized model must be initialized with either selected'
                    ' features or a model path')

            self.selected_features_lists = selected_features_lists
            extractor_groups = self._get_feature_extractor_groups()
            self.featurizers = [
                self._make_featurizer(extractors, selected_features_list, i)
                for i, (selected_features_list, extractors)
                in enumerate(zip(self.selected_features_lists,
                                 extractor_groups))]


    def _make_featurizer(self, extractors, featurizer_params, featurizer_index):
        return Featurizer(extractors, featurizer_params, self.save_featurized)

    @classmethod
    def _get_feature_extractor_groups(klass):
        raise NotImplementedError

    def reset(self):
        super(FeaturizedModelBase, self).reset()
        for featurizer in self.featurizers:
            featurizer.reset()


class FeaturizedModel(FeaturizedModelBase):
    def __init__(self, selected_features=None, model_path=None,
                 save_featurized=False, *args, **kwargs):
        super(FeaturizedModel, self).__init__(
            selected_features_lists=[selected_features], model_path=model_path,
            save_featurized=save_featurized, *args, **kwargs)
        self.featurizer = self.featurizers[0]

    @classmethod
    def _get_feature_extractor_groups(klass):
        return ['.'.join([klass.__module__, klass.__name__,
                          'all_feature_extractors'])]

    # Subclasses should override this class-level variable to include actual
    # feature extractor objects.
    all_feature_extractors = []


class MultiplyFeaturizedModel(FeaturizedModelBase):
    def __init__(self, model_path=None, selected_features=None,
                 *args, **kwargs):
        selected_features_lists = []
        selected_features = set(selected_features)
        for extractor_group in self._get_feature_extractor_groups():
            group_selected_names = [e.name for e in extractor_group
                                    if e.name in selected_features]
            selected_features_lists.append(group_selected_names)

        super(MultiplyFeaturizedModel, self).__init__(
            model_path=model_path,
            selected_features_lists=selected_features_lists, *args, **kwargs)

    @classmethod
    def _get_feature_extractor_groups(klass):
        return klass.feature_extractor_groups
    
    # Subclasses should override this class-level variable to include lists of
    # actual feature extractor objects.
    feature_extractor_groups = []
    

class ClassifierModel(FeaturizedModel):
    ''' Wraps a scikit-learn classifier for use in a pipeline stage. '''

    def __init__(self, classifier, *args, **kwargs):
        """
        Note that classifier must support the fit and predict methods in the
        style of scikit-learn.
        """
        self.classifier = classifier
        super(ClassifierModel, self).__init__(*args, **kwargs)

    def _train_model(self, instances):
        # print "Featurizing", len(instances), "instances"
        logging.debug("Registering features...")
        self.featurizer.register_features_from_instances(instances)
        logging.debug('Done registering features.')

        features = self.featurizer.featurize(instances)
        labels = self._get_gold_labels(instances)

        '''
        print("Featurized train:")
        strings = []
        for i in range(len(instances)):
            d = self.featurizer.matrow2dict(features, i)
            label = bool(labels[i])
            strings.append('{%s} -> %s' % (', '.join(
                ['%s: %s' % (key, d[key]) for key in sorted(d.keys())]), label))
        strings.sort()
        print '\n'.join(strings)
        # '''

        logging.debug('Fitting classifier...')
        self.classifier.fit(features, labels)
        logging.debug('Done fitting classifier.')
        return self.featurizer.feature_name_dictionary

    def test(self, instances):
        features = self.featurizer.featurize(instances)
        gold_labels = self._get_gold_labels(instances)
        labels = self.classifier.predict(features)

        if self.save_featurized:
            self.gold_labels = gold_labels
            self.labels = labels
            self.raw_instances = instances

        '''
        print("Featurized test:")
        strings = []
        for i in range(len(instances)):
            d = self.featurizer.matrow2dict(features, i)
            label = bool(labels[i])
            gold_label = bool(gold_labels[i])
            strings.append('{%s} -> %s / %s' % (', '.join(
                ['%s: %s' % (key, d[key]) for key in sorted(d.keys())]), label,
                gold_label))
        strings.sort()
        print '\n'.join(strings)
        # '''

        # logging.debug('%d data points' % len(gold_labels))
        # logging.debug('Raw classifier performance:')
        # logging.debug('\n' + str(diff_binary_vectors(labels, gold_labels)))
        return labels

    def _get_gold_labels(self, instances):
        raise NotImplementedError


class ClassBalancingClassifierWrapper(BaseEstimator):
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
                indices = np.random.choice(label_row_indices, counts_to_add[j])
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

    def predict_proba(self, data):
        return self.classifier.predict_proba(data)


# TODO: remove in favor of sklearn's DummyClassifier?
class MajorityClassClassifier(ClassifierModel):
    def __init__(self):
        self.decision = None

    def reset(self):
        self.decision = None

    def _train_model(self, instances):
        labels = self._get_gold_labels(instances)
        label_counts = [0, 0]
        for label in labels:
            label_counts[bool(label)] += 1
        self.decision = label_counts[True] > label_counts[False]

    def test(self, instances):
        return [self.decision for _ in instances]


class SimpleModel(Model):
    '''
    A model that simply runs an operation on each instance, with the operation
    encapsulated as a single function. The function should be entirely
    self-contained; its return value will be ignored.
    '''
    def __init__(self, operation):
        self.operation = operation

    def test(self, instances):
        for instance in instances:
            self.operation(instance)
        # No return value -> no labeling will be done

    def save(self, filepath):
        # Nothing to save
        pass

    def _load_model(self, filepath):
        # Nothing to load
        pass
