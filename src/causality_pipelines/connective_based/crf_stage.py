from copy import copy
from gflags import DEFINE_list, DEFINE_string, DEFINE_bool, DEFINE_integer, FLAGS, DuplicateFlagError, DEFINE_enum
import logging
import numpy as np

from causality_pipelines.connective_based import PossibleCausation
from iaa import CausalityMetrics
from pipeline import Stage
from pipeline.models import CRFModel
from pipeline.feature_extractors import FeatureExtractor
from util import Enum

try:
    DEFINE_list('arg_label_features',
                ['lemma', 'pos', 'is_connective', # 'conn_parse_dist',
                 'conn_parse_path', 'lexical_conn_dist', 'in_parse_tree',
                 'pattern', 'pattern+conn_parse_path', 'conn_rel_pos'],
                'Features for the argument-labeling CRF')
    DEFINE_string('arg_label_model_path', 'arg-labeler-crf.model',
                  'Path to save the argument-labeling CRF model to')
    DEFINE_bool('arg_label_log_differences', False,
                'Whether to log differences between gold and predicted results'
                ' for each round of argument labeling evaluation')
    DEFINE_integer('arg_label_max_dep_path_len', 4,
                   "Maximum number of dependency path steps to allow before"
                   " just making the value 'LONG-RANGE'")
    DEFINE_enum('arg_label_training_alg', 'lbfgs',
                ['lbfgs', 'l2sgd', 'ap', 'pa', 'arow'],
                'Algorithm for training argument labeling CRF')
except DuplicateFlagError as e:
    logging.warn('Ignoring redefinition of flag %s' % e.flagname)


class ArgumentLabelerModel(CRFModel):
    CAUSE_LABEL = 'Cause'
    EFFECT_LABEL = 'Effect'

    def __init__(self, training_algorithm, training_params):
        super(ArgumentLabelerModel, self).__init__(
            PossibleCausation, FLAGS.arg_label_model_path,
            self.FEATURE_EXTRACTOR_MAP, FLAGS.arg_label_features,
            training_algorithm, training_params)

    def _sequences_for_part(self, part, is_train):
        # part for this model is a PossibleCausation.
        observations = part.sentence.tokens
        if is_train:
            labels = ['None'] * len(observations)
            if part.true_causation_instance.cause:
                for cause_token in part.true_causation_instance.cause:
                    labels[cause_token.index] = self.CAUSE_LABEL
            if part.true_causation_instance.effect:
                for effect_token in part.true_causation_instance.effect:
                    labels[effect_token.index] = self.EFFECT_LABEL
        else: # testing time
            labels = None
        return observations, labels
    
    def _label_part(self, part, crf_labels):
        part.cause_tokens = []
        part.effect_tokens = []
        for token, label in zip(part.sentence.tokens, crf_labels):
            if label == self.CAUSE_LABEL:
                part.cause_tokens.append(token)
            elif label == self.EFFECT_LABEL:
                part.effect_tokens.append(token)

    @staticmethod
    def get_connective_parse_distance(observation):
        sentence = observation.part.sentence
        _, closest_connective_distance = sentence.get_closest_of_tokens(
            observation.observation, observation.part.connective_tokens)
        return closest_connective_distance

    @staticmethod
    def get_connective_parse_path(observation):
        word = observation.observation
        sentence = observation.part.sentence
        closest_connective_token, _ = sentence.get_closest_of_tokens(
            word, observation.part.connective_tokens)
        if closest_connective_token is None:
            return 'NO_PATH'

        deps = sentence.extract_dependency_path(word, closest_connective_token,
                                                False)
        if len(deps) > FLAGS.arg_label_max_dep_path_len:
            return 'LONG-RANGE'
        else:
            return str(deps)

    RELATIVE_POSITIONS = Enum(['Before', 'Overlapping', 'After'])
    @staticmethod
    def get_connective_relative_position(observation):
        word = observation.observation
        sentence = observation.part.sentence
        closest_connective_token, _ = sentence.get_closest_of_tokens(
            word, observation.part.connective_tokens, False) # lexically closest
        if word.index < closest_connective_token.index:
            return ArgumentLabelerModel.RELATIVE_POSITIONS.Before
        elif word.index > closest_connective_token.index:
            return ArgumentLabelerModel.RELATIVE_POSITIONS.After
        else:
            return ArgumentLabelerModel.RELATIVE_POSITIONS.Overlapping

    class LexicalDistanceFeatureExtractor(FeatureExtractor):
        ABS_DIST_NAME = 'absdist'
        DIRECTED_DIST_NAME = 'dirdist'

        def __init__(self, name):
            self.name = name
            self.feature_type = self.FeatureTypes.Numerical

        def extract_subfeature_names(self, parts):
            return [self.ABS_DIST_NAME, self.DIRECTED_DIST_NAME]

        def extract(self, observation):
            word = observation.observation
            # TODO: make this function return a signed value?
            min_distance = np.inf
            min_abs_distance = np.inf
            for connective_token in observation.part.connective_tokens:
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
# be a sequence of Tokens. observation.part will be a PossibleCausation.
FEATURE_EXTRACTORS = [
    FeatureExtractor(
        'lemma', lambda observation: observation.observation.lemma),
    FeatureExtractor(
        'pos', lambda observation: observation.observation.pos),
    FeatureExtractor(
        'is_connective',
        lambda observation: (observation.observation in
                             observation.part.connective_tokens)),
    FeatureExtractor(
        'conn_parse_path', ArgumentLabelerModel.get_connective_parse_path),
    FeatureExtractor(
        'conn_parse_dist', ArgumentLabelerModel.get_connective_parse_distance,
        FeatureExtractor.FeatureTypes.Numerical),
    ArgumentLabelerModel.LexicalDistanceFeatureExtractor('lexical_conn_dist'),
    FeatureExtractor('in_parse_tree',
                     lambda observation: (observation.part.sentence.get_depth(
                                            observation.observation) < np.inf)),
    FeatureExtractor('pattern',
                     lambda observation: observation.part.matching_pattern),
    FeatureExtractor(
        'pattern+conn_parse_path',
        lambda observation: (
            observation.part.matching_pattern,
            ArgumentLabelerModel.get_connective_parse_path(observation))),
    FeatureExtractor('conn_rel_pos',
                     ArgumentLabelerModel.get_connective_relative_position)
]

ArgumentLabelerModel.FEATURE_EXTRACTOR_MAP = {
    extractor.name: extractor for extractor in FEATURE_EXTRACTORS}


class ArgumentLabelerStage(Stage):
    def __init__(self, name, training_algorithm=None, training_params={}):
        if training_algorithm is None:
            training_algorithm = FLAGS.arg_label_training_alg
        super(ArgumentLabelerStage, self).__init__(
            name=name,
            models=[ArgumentLabelerModel(training_algorithm, training_params)])
        self.print_test_instances = FLAGS.regex_print_test_instances

    def _extract_parts(self, sentence, is_train):
        if is_train:
            # Filter to possible causations for which we can actually extract
            # the correct labels, i.e., gold-standard causations.
            return [possible_causation for possible_causation
                    in sentence.possible_causations
                    if possible_causation.true_causation_instance]
        else:
            return sentence.possible_causations

    def _decode_labeled_parts(self, sentence, labeled_parts):
        sentence.causation_instances = []
        for possible_causation in labeled_parts:
            sentence.add_causation_instance(
                connective=possible_causation.connective_tokens,
                cause=possible_causation.cause_tokens,
                effect=possible_causation.effect_tokens)

    CONSUMED_ATTRIBUTES = ['possible_causations']

    PERMISSIVE_KEY = 'Allowing partial matches'
    STRICT_KEY = 'Not allowing partial matches'
    @staticmethod
    def aggregate_eval_results(results_list):
        permissive = CausalityMetrics.aggregate(
            [result_dict[ArgumentLabelerStage.PERMISSIVE_KEY]
             for result_dict in results_list])
        strict = CausalityMetrics.aggregate(
            [result_dict[ArgumentLabelerStage.STRICT_KEY]
             for result_dict in results_list])
        return {ArgumentLabelerStage.PERMISSIVE_KEY: permissive,
                ArgumentLabelerStage.STRICT_KEY: strict}

    def _begin_evaluation(self):
        self._predicted_sentences = []

    def _prepare_for_evaluation(self, sentences):
        # Copy over the existing ParsedSentence objects, with their pointers to
        # their causation instance lists. That way we can pass them to
        # CausalityMetrics as gold sentences.
        # We also need to tell each CausationInstance that it has a new parent,
        # or bad things will happen when we try to run IAA for evaluation.
        self._gold_sentences = []
        for sentence in sentences:
            new_sentence = copy(sentence)
            self._gold_sentences.append(new_sentence)
            for causation_instance in new_sentence.causation_instances:
                causation_instance.source_sentence = new_sentence

    def _evaluate(self, sentences):
        self._predicted_sentences.extend(sentences)
        
    def _complete_evaluation(self):
        with_partial = CausalityMetrics(
            self._gold_sentences, self._predicted_sentences, True,
            FLAGS.arg_label_log_differences,
            compare_degrees=False, compare_types=False)
        without_partial = CausalityMetrics(
            self._gold_sentences, self._predicted_sentences, False,
            FLAGS.arg_label_log_differences,
            compare_degrees=False, compare_types=False)

        # TODO: actually log differences here

        del self._gold_sentences
        del self._predicted_sentences
        return {self.PERMISSIVE_KEY: with_partial,
                self.STRICT_KEY: without_partial}
