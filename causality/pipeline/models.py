""" Define standard machine-learned model framework for pipelines. """

import numpy as np
import warnings

class Model(object):
    def __init__(self, part_type):
        self.part_type = part_type

    def train(self, destination_path):
        raise NotImplementedError

    def test(self, destination_path):
        raise NotImplementedError


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
        super(FeaturizedModel, self).__init__(part_type)
        self.selected_features = selected_features
        self.feature_name_dictionary = FeaturizedModel.NameDictionary()
        self.feature_extractor_map = feature_extractor_map

    def train(self, parts):
        # Build feature name dictionary if not already present.
        if not self.feature_name_dictionary:
            feature_values = {}
            for feature_name in self.selected_features:
                is_boolean, extractor_fn = (
                    self.feature_extractor_map[feature_name])
                if not is_boolean:
                    self.feature_name_dictionary.insert(feature_name)
                else:
                    value_set = [extractor_fn(part) for part in parts]
                    feature_values[feature_name] = value_set

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
        self.classifier.fit(features, labels)

    def _featurized_test(self, parts):
        features, _ = self._featurize(parts)
        return self.classifier.predict(features)

    def _featurize(self, parts):
        relevant_parts = [part for part in parts if isinstance(part,
                                                               self.part_type)]
        features = np.zeros((len(relevant_parts),
                             len(self.feature_name_dictionary)))
        labels = np.array([part.label for part in relevant_parts])

        for part, row_ref in zip(relevant_parts, features):
            for feature_name in self.selected_features:
                is_boolean, extractor_fn = (
                    self.feature_extractor_map[feature_name])
                feature_value = extractor_fn(part)
                if is_boolean:
                    feature_name = self.get_boolean_feature_name(feature_name,
                                                                 feature_value)
                    feature_value = 1.0
                try:
                    row_ref[self.feature_name_dictionary[feature_name]
                            ] = feature_value
                except KeyError:
                    warnings.warn('Ignoring unknown feature: %s' % feature_name)

        return features, labels

class ClassifierPart(object):
    def __init__(self, instance, label):
        self.label = int(label)
        self.instance = instance
