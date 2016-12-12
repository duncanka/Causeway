from gflags import (DEFINE_list, DEFINE_bool, DEFINE_integer, FLAGS,
                    DuplicateFlagError, DEFINE_enum)
from itertools import chain
import logging
import numpy as np
import os

from causeway import IAAEvaluator, RELATIVE_POSITIONS
from nlpypline.pipeline.models.structured import CRFModel
from nlpypline.pipeline.featurization import FeatureExtractor, SetValuedFeatureExtractor
from nlpypline.pipeline import Stage

try:
    DEFINE_list('arg_label_features',
                ['lemma', 'pos', 'is_connective', # 'conn_parse_dist',
                 'conn_parse_path', 'lexical_conn_dist', 'in_parse_tree',
                 'pattern', 'pattern+conn_parse_path', 'conn_rel_pos',
                 'is_alnum'],
                'Features for the argument-labeling CRF')
    DEFINE_integer('arg_label_max_dep_path_len', 4,
                   "Maximum number of dependency path steps to allow before"
                   " just making the value 'LONG-RANGE'")
    DEFINE_enum('arg_label_training_alg', 'lbfgs',
                ['lbfgs', 'l2sgd', 'ap', 'pa', 'arow'],
                'Algorithm for training argument labeling CRF')
    DEFINE_bool('arg_label_save_crf_info', False,
                'Whether to read in and save an accessible version of the CRF'
                ' model parameters in the model (useful for debugging)')
except DuplicateFlagError as e:
    logging.warn('Ignoring flag redefinitions; assuming module reload')


class ArgumentLabelerModel(CRFModel):
    CAUSE_LABEL = 'Cause'
    EFFECT_LABEL = 'Effect'
    NONE_LABEL = 'None'

    def __init__(self, training_algorithm, training_params, *args, **kwargs):
        super(ArgumentLabelerModel, self).__init__(
            selected_features=FLAGS.arg_label_features,
            training_algorithm=training_algorithm,
            training_params=training_params, *args, **kwargs)

    def _sequence_for_instance(self, possible_causation, is_train):
        return possible_causation.sentence.tokens[1:] # all tokens but ROOT

    def _get_gold_labels(self, crf_part_type, crf_parts):
        labels = [self.NONE_LABEL] * len(crf_parts)
        for i, crf_part in enumerate(crf_parts):
            pc = crf_part.instance
            true_instance = pc.true_causation_instance
            token = crf_part.observation
            if true_instance.cause and token in true_instance.cause:
                labels[i] = self.CAUSE_LABEL
            elif true_instance.effect and token in true_instance.effect:
                labels[i] = self.EFFECT_LABEL

        return labels

    @staticmethod
    def get_connective_parse_distance(observation):
        sentence = observation.instance.sentence
        _, closest_connective_distance = sentence.get_closest_of_tokens(
            observation.observation, observation.instance.connective)
        return closest_connective_distance

    @staticmethod
    def get_connective_parse_path(observation):
        word = observation.observation
        sentence = observation.instance.sentence
        closest_connective_token, _ = sentence.get_closest_of_tokens(
            word, observation.instance.connective)
        if closest_connective_token is None:
            return 'NO_PATH'

        deps = sentence.extract_dependency_path(closest_connective_token, word,
                                                False)
        if len(deps) > FLAGS.arg_label_max_dep_path_len:
            return 'LONG-RANGE'
        else:
            return str(deps)

    @staticmethod
    def get_connective_relative_position(observation):
        word = observation.observation
        sentence = observation.instance.sentence
        closest_connective_token, _ = sentence.get_closest_of_tokens(
            word, observation.instance.connective, False) # lexically closest
        if word.index < closest_connective_token.index:
            return RELATIVE_POSITIONS.Before
        elif word.index > closest_connective_token.index:
            return RELATIVE_POSITIONS.After
        else:
            return RELATIVE_POSITIONS.Overlapping

    class LexicalDistanceFeatureExtractor(FeatureExtractor):
        ABS_DIST_NAME = 'absdist'
        DIRECTED_DIST_NAME = 'dirdist'

        def __init__(self, name):
            self.name = name
            self.feature_type = self.FeatureTypes.Numerical

        def extract_subfeature_names(self, instances):
            return [self.ABS_DIST_NAME, self.DIRECTED_DIST_NAME]

        def extract(self, observation):
            word = observation.observation
            min_distance = np.inf
            min_abs_distance = np.inf
            for connective_token in observation.instance.connective:
                new_distance = connective_token.index - word.index
                new_abs_distance = abs(new_distance)
                if new_abs_distance < min_abs_distance:
                    min_distance = new_distance
                    min_abs_distance = new_abs_distance
            return {self.ABS_DIST_NAME: min_abs_distance,
                    self.DIRECTED_DIST_NAME: min_distance}

# Because this is a CRF model operating on sequences of tokens, the input to
# each feature extractor will be a CRFModel.ObservationWithContext.
# observation.observation will be a Token object, and observation.sequence will
# be a sequence of Tokens. observation.instance will be a PossibleCausation.
ArgumentLabelerModel.all_feature_extractors = [
    FeatureExtractor(
        'lemma', lambda observation: observation.observation.lemma),
    FeatureExtractor(
        'pos', lambda observation: observation.observation.pos),
    FeatureExtractor(
        'is_connective',
        lambda observation: (observation.observation in
                             observation.instance.connective),
        FeatureExtractor.FeatureTypes.Binary),
    FeatureExtractor(
        'conn_parse_path', ArgumentLabelerModel.get_connective_parse_path),
    FeatureExtractor(
        'conn_parse_dist', ArgumentLabelerModel.get_connective_parse_distance,
        FeatureExtractor.FeatureTypes.Numerical),
    ArgumentLabelerModel.LexicalDistanceFeatureExtractor('lexical_conn_dist'),
    FeatureExtractor(
        'in_parse_tree',
        lambda observation: (observation.instance.sentence.get_depth(
                                observation.observation) < np.inf),
                     FeatureExtractor.FeatureTypes.Binary),
    # Use repr to get around issues with ws at the end of feature names (breaks
    # CRFSuite dump parser)
    SetValuedFeatureExtractor(
        'pattern',
        lambda observation: [repr(pattern) for pattern in
                             observation.instance.matching_patterns]),
    SetValuedFeatureExtractor(
        'pattern+conn_parse_path',
        lambda observation: [ # Use quotes to work around stupid ws issue
            '%s / "%s"' % (
                pattern, ArgumentLabelerModel.get_connective_parse_path(
                            observation))
            for pattern in observation.instance.matching_patterns]),
    FeatureExtractor('conn_rel_pos',
                     ArgumentLabelerModel.get_connective_relative_position),
    FeatureExtractor('is_alnum', lambda observation: (
                                    observation.observation.lemma.isalnum()),
                     FeatureExtractor.FeatureTypes.Binary)
]


class ArgumentLabelerEvaluator(IAAEvaluator):
    def evaluate(self, document, original_document, possible_causations,
                 original_pcs):
        # The causations are attached to the sentences in the documents, so we
        # don't even have to care about them -- we can just grab the sentences
        # straight from the documents.
        return super(ArgumentLabelerEvaluator, self).evaluate(
            document, original_document, document.sentences,
            original_document.sentences)


class ArgumentLabelerStage(Stage):
    def __init__(self, name, training_algorithm=None, training_params={}):
        if training_algorithm is None:
            training_algorithm = FLAGS.arg_label_training_alg
        if not os.path.isdir(FLAGS.models_dir):
            os.makedirs(FLAGS.models_dir)
        model_file_path = os.path.join(FLAGS.models_dir, "%s.model" % name)
        model = ArgumentLabelerModel(training_algorithm, training_params,
                                     model_file_path=model_file_path)
        super(ArgumentLabelerStage, self).__init__(name, model)

    def _extract_instances(self, document, is_train, is_original):
        # No possible causations in the gold-standard data. But it doesn't
        # matter; evaluation will be done by comparing gold to possible, not
        # possible to possible. Just return no instances.
        if is_original:
            return []

        if is_train:
            # Filter to possible causations for which we can actually
            # extract the correct labels, i.e., gold-standard causations.
            pcs = [[possible_causation for possible_causation
                    in sentence.possible_causations
                    if possible_causation.true_causation_instance]
                   for sentence in document]
        else:
            pcs = [sentence.possible_causations for sentence in document]
        return list(chain.from_iterable(pcs))

    def _label_instance(self, document, possible_causation, predicted_labels):
        sentence = possible_causation.sentence
        possible_causation.cause = []
        possible_causation.effect = []
        # Labels exclude ROOT token.
        for token, label in zip(sentence.tokens[1:], predicted_labels):
            if label == ArgumentLabelerModel.CAUSE_LABEL:
                possible_causation.cause.append(token)
            elif label == ArgumentLabelerModel.EFFECT_LABEL:
                possible_causation.effect.append(token)

        if not possible_causation.cause: possible_causation.cause = None
        if not possible_causation.effect: possible_causation.effect = None

    '''
    def _document_complete(self, document):
        for sentence in document:
            # print "Unfiltered:", sentence.possible_causations
            sentence.possible_causations = [
                pc for pc in sentence.possible_causations
                if pc.cause and pc.effect]
            # print "Filtered:", sentence.possible_causations
    '''

    def _make_evaluator(self):
        return ArgumentLabelerEvaluator(
            False, False, FLAGS.args_print_test_instances, True, True,
            'possible_causations')
