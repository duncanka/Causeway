""" Define standard machine-learned model framework for pipelines. """

import gflags
import logging
import numpy as np

from util.metrics import diff_binary_vectors

try:
    gflags.DEFINE_bool(
        'rebalance_stochastically', False,
        'Rebalance classes by stochastically choosing samples to replicate')
except gflags.DuplicateFlagError as e:
    logging.warn('Ignoring redefinition of flag %s' % e.flagname)


class Model(object):
    def __init__(self, part_type):
        self.part_type = part_type

    def train(self, destination_path):
        raise NotImplementedError

    def test(self, destination_path):
        raise NotImplementedError

class TrainableFeatureExtractor(object):
    def __init__(self, feature_training_data_extractor,
                 feature_extractor_creator):
        self.feature_training_data_extractor = feature_training_data_extractor
        self.feature_extractor_creator = feature_extractor_creator
        self.subfeature_extractor_map = None

    def train(self, parts):
        extracted_data = self.feature_training_data_extractor(parts)
        self.subfeature_extractor_map = self.feature_extractor_creator(
            extracted_data)
        return extracted_data

class FeaturizedModel(Model):
    class NameDictionary(object):
        def __init__(self):
            self.names_to_ids = {}
            self.ids_to_names = []

        def insert(self, entry):
            if not self.names_to_ids.has_key(entry):
                self.names_to_ids[entry] = len(self.names_to_ids)
                self.ids_to_names.append(entry)

        def __getitem__(self, entry):
            if isinstance(entry, int):
                return self.ids_to_names[entry]
            else: # it's a string name
                return self.names_to_ids[entry]

        def __len__(self):
            return len(self.names_to_ids)

        def __contains__(self, entry):
            return self.names_to_ids.has_key(entry)

    @staticmethod
    def get_boolean_feature_name(base_name, value):
        return '%s=%s' % (base_name, value)

    def __init__(self, part_type, feature_extractor_map, selected_features):
        """
        part_type is the class object corresponding to the part type this model
        is for.
        feature_extractor_map is a map from feature names to
        (feature_is_boolean, feature_extractor_function) tuples.
        selected_features is a list of names of features to extract.
        """
        super(FeaturizedModel, self).__init__(part_type)
        self.selected_features = selected_features
        self.feature_name_dictionary = FeaturizedModel.NameDictionary()
        self.feature_extractor_map = feature_extractor_map
        self.feature_training_data = {}

    def train(self, parts):
        # Build feature name dictionary. (Unfortunately, this means we featurize
        # everything twice, but I can't think of a cleverer way.)
        feature_values = {}

        def insert_names(feature_name, is_categorical, feature_extractor):
            if is_categorical:
                value_set = set([feature_extractor(part) for part in parts])
                feature_values[feature_name] = value_set
            else:
                self.feature_name_dictionary.insert(feature_name)

        for feature_name in self.selected_features:
            is_categorical, extractor = (
                self.feature_extractor_map[feature_name])

            if isinstance(extractor, TrainableFeatureExtractor):
                self.feature_training_data[feature_name] = (
                    extractor.train(parts))
                for subfeature_name, subfeature_extractor in (
                    extractor.subfeature_extractor_map.iteritems()):
                    insert_names(subfeature_name, is_categorical,
                                 subfeature_extractor)
            else:
                insert_names(feature_name, is_categorical, extractor)

        # All the ones we logged feature values for were the boolean ones.
        # Now we register all the corresponding feature names.
        for base_name, value_set in feature_values.iteritems():
            for value in value_set:
                self.feature_name_dictionary.insert(
                    self.get_boolean_feature_name(base_name, value))

        self._featurized_train(parts)

    def test(self, parts):
        assert self.feature_name_dictionary, (
            "Feature name dictionary must be populated either by training or by"
            " loading a model")
        self._featurized_test(parts)

    def _featurized_train(self, parts):
        raise NotImplementedError

    def _featurized_test(self, parts):
        raise NotImplementedError


class ClassifierModel(FeaturizedModel):
    def __init__(self, part_type, feature_extractor_map, selected_features,
                 classifier):
        """
        Note that classifier must support the fit and predict methods in the
        style of scikit-learn.
        """
        super(ClassifierModel, self).__init__(part_type, feature_extractor_map,
                                              selected_features)
        self.classifier = classifier

    def _featurized_train(self, parts):
        features, labels = self._featurize(parts)
        logging.info('Fitting classifier...')
        self.classifier.fit(features, labels)
        logging.info('Done fitting classifier.')

    def _featurized_test(self, parts):
        features, old_labels = self._featurize(parts)
        labels = self.classifier.predict(features)
        for part, label in zip(parts, labels):
            part.label = label
        logging.debug('%d data points' % len(old_labels))
        logging.debug('Raw classifier performance:')
        logging.debug('\n' + str(diff_binary_vectors(labels, old_labels)))
        '''
        for part, label, old_label in zip(parts[:1000], labels[:1000], old_labels[:1000]):
            if label != old_label:
                heads = (part.head_token_1, part.head_token_2)
                if label and not old_label:
                    print 'False positive: %s (%s)' % (
                        heads, part.instance.original_text)
                elif old_label and not label:
                    print 'False negative: %s (%s)' % (
                        heads, part.instance.original_text)
        #'''


    def _featurize(self, parts):
        logging.info('Featurizing...')
        relevant_parts = [part for part in parts if isinstance(part,
                                                               self.part_type)]
        features = np.zeros((len(relevant_parts),
                             len(self.feature_name_dictionary)))
        labels = np.array([part.label for part in relevant_parts])

        for part, row_ref in zip(relevant_parts, features):
            for feature_name in self.selected_features:
                is_boolean, extractor = (
                    self.feature_extractor_map[feature_name])

                def insert_value(feature_name, feature_extractor):
                    feature_value = feature_extractor(part)
                    if is_boolean:
                        feature_name = self.get_boolean_feature_name(feature_name,
                                                                     feature_value)
                        feature_value = 1.0
                    try:
                        row_ref[self.feature_name_dictionary[feature_name]
                                ] = feature_value
                    except KeyError:
                        logging.debug('Ignoring unknown feature: %s' % feature_name)

                if isinstance(extractor, TrainableFeatureExtractor):
                    for subfeature_name, subfeature_extractor in (
                        extractor.subfeature_extractor_map.iteritems()):
                        insert_value(subfeature_name, subfeature_extractor)
                else:
                    insert_value(feature_name, extractor)

        logging.info('Done featurizing.')
        return features, labels

class ClassifierPart(object):
    def __init__(self, instance, label):
        self.label = int(label)
        self.instance = instance

class ClassBalancingModelWrapper(object):
    def __init__(self, classifier, ratio=float('inf')):
        self.classifier = classifier
        self.ratio = ratio

    @staticmethod
    def rebalance(data, labels, ratio=float('inf')):
        """
        ratio indicates the maximum ratio by which any class is allowed to
        increase.
        """
        if ratio <= 1.0: # No increase
            return data, labels

        # Based on http://stackoverflow.com/a/23392678/4044809
        label_set, label_indices, label_counts = np.unique(
            labels, return_inverse=True, return_counts=True)
        max_count = label_counts.max()
        counts_to_add = [min(max_count - current_count, ratio * current_count)
                         for current_count in label_counts]
        counts_to_add = [int(round(count)) for count in counts_to_add]
        rows_to_add = np.sum(counts_to_add)
        rebalanced_data = np.empty((rows_to_add, data.shape[1]), data.dtype)
        rebalanced_labels = np.empty((rows_to_add,), labels.dtype)

        slices = np.concatenate(([0], np.cumsum(counts_to_add)))
        for j in xrange(len(label_set)):
            label_row_indices = np.where(label_indices==j)[0]
            if gflags.FLAGS.rebalance_stochastically:
                indices = np.random.choice(label_row_indices,
                                           counts_to_add[j])
            else:
                full_repetitions = counts_to_add[j] / label_row_indices.shape[0]
                indices = np.tile(label_row_indices, (full_repetitions,))
                still_needed = indices.shape[0] - counts_to_add[j]
                if still_needed:
                    indices = np.concatenate(indices,
                                             label_row_indices[:still_needed])

            if label_indices.shape[0]: # only bother if there are > 0 indices
                rebalanced_data[slices[j]:slices[j+1]] = data[indices]
                rebalanced_labels[slices[j]:slices[j+1]] = labels[indices]
        rebalanced_data = np.vstack((data, rebalanced_data))
        rebalanced_labels = np.concatenate((labels, rebalanced_labels))
        return (rebalanced_data, rebalanced_labels)

    def fit(self, data, labels):
        rebalanced_data, rebalanced_labels = self.rebalance(
            data, labels, self.ratio)
        return self.classifier.fit(rebalanced_data, rebalanced_labels)

    def predict(self, data):
        return self.classifier.predict(data)
