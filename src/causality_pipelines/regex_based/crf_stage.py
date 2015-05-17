from gflags import DEFINE_list, DEFINE_string, DEFINE_bool, DEFINE_integer, FLAGS, DuplicateFlagError, DEFINE_enum
import logging
import numpy as np

from causality_pipelines import PossibleCausation, IAAEvaluator
from pipeline.models import CRFModel
from pipeline.feature_extractors import FeatureExtractor, SetValuedFeatureExtractor
from util import Enum
from pipeline import Stage

try:
    DEFINE_list('arg_label_features',
                ['lemma', 'pos', 'is_connective', # 'conn_parse_dist',
                 'conn_parse_path', 'lexical_conn_dist', 'in_parse_tree',
                 'pattern', 'pattern+conn_parse_path', 'conn_rel_pos',
                 'is_alnum'],
                'Features for the argument-labeling CRF')
    DEFINE_string('arg_label_model_path', '../arg-labeler-crf.model',
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
    DEFINE_bool('arg_label_save_crf_info', False,
                'Whether to read in and save an accessible version of the CRF'
                ' model parameters in the model (useful for debugging)')
except DuplicateFlagError as e:
    logging.warn('Ignoring redefinition of flag %s' % e.flagname)


class ArgumentLabelerModel(CRFModel):
    CAUSE_LABEL = 'Cause'
    EFFECT_LABEL = 'Effect'

    def __init__(self, training_algorithm, training_params):
        super(ArgumentLabelerModel, self).__init__(
            PossibleCausation, FLAGS.arg_label_model_path,
            self.FEATURE_EXTRACTORS, FLAGS.arg_label_features,
            training_algorithm, training_params, FLAGS.arg_label_save_crf_info)

    def _sequences_for_part(self, part, is_train):
        # part for this model is a PossibleCausation.
        observations = part.sentence.tokens[1:] # Exclude ROOT (hence -1s below)
        if is_train:
            labels = ['None'] * len(observations)
            if part.true_causation_instance.cause:
                for cause_token in part.true_causation_instance.cause:
                    labels[cause_token.index - 1] = self.CAUSE_LABEL
            if part.true_causation_instance.effect:
                for effect_token in part.true_causation_instance.effect:
                    labels[effect_token.index - 1] = self.EFFECT_LABEL
        else: # testing time
            labels = None
        return observations, labels
    
    def _label_part(self, part, crf_labels):
        part.cause = []
        part.effect = []
        # Labels exclude ROOT token.
        for token, label in zip(part.sentence.tokens[1:], crf_labels):
            if label == self.CAUSE_LABEL:
                part.cause.append(token)
            elif label == self.EFFECT_LABEL:
                part.effect.append(token)

        if not part.cause:
            part.cause = None
        if not part.effect:
            part.effect = None

    @staticmethod
    def get_connective_parse_distance(observation):
        sentence = observation.part.sentence
        _, closest_connective_distance = sentence.get_closest_of_tokens(
            observation.observation, observation.part.connective)
        return closest_connective_distance

    @staticmethod
    def get_connective_parse_path(observation):
        word = observation.observation
        sentence = observation.part.sentence
        closest_connective_token, _ = sentence.get_closest_of_tokens(
            word, observation.part.connective)
        if closest_connective_token is None:
            return 'NO_PATH'

        deps = sentence.extract_dependency_path(closest_connective_token, word,
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
            word, observation.part.connective, False) # lexically closest
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
            min_distance = np.inf
            min_abs_distance = np.inf
            for connective_token in observation.part.connective:
                new_distance = connective_token.index - word.index
                new_abs_distance = abs(new_distance)
                if new_abs_distance < min_abs_distance:
                    min_distance = new_distance
                    min_abs_distance = new_abs_distance
            return {self.ABS_DIST_NAME: min_abs_distance,
                    self.DIRECTED_DIST_NAME: min_distance}

    # We can't initialize this properly yet because we don't have access to the
    # class' static methods to define the list.
    FEATURE_EXTRACTORS = []

# Because this is a CRF model operating on sequences of tokens, the input to
# each feature extractor will be a CRFModel.ObservationWithContext.
# observation.observation will be a Token object, and observation.sequence will
# be a sequence of Tokens. observation.part will be a PossibleCausation.
ArgumentLabelerModel.FEATURE_EXTRACTORS = [
    FeatureExtractor(
        'lemma', lambda observation: observation.observation.lemma),
    FeatureExtractor(
        'pos', lambda observation: observation.observation.pos),
    FeatureExtractor(
        'is_connective',
        lambda observation: (observation.observation in
                             observation.part.connective),
        FeatureExtractor.FeatureTypes.Numerical),
    FeatureExtractor(
        'conn_parse_path', ArgumentLabelerModel.get_connective_parse_path),
    FeatureExtractor(
        'conn_parse_dist', ArgumentLabelerModel.get_connective_parse_distance,
        FeatureExtractor.FeatureTypes.Numerical),
    ArgumentLabelerModel.LexicalDistanceFeatureExtractor('lexical_conn_dist'),
    FeatureExtractor('in_parse_tree',
                     lambda observation: (observation.part.sentence.get_depth(
                                            observation.observation) < np.inf),
                     FeatureExtractor.FeatureTypes.Numerical),
    # Use repr to get around issues with ws at the end of feature names (breaks
    # CRFSuite dump parser)
    SetValuedFeatureExtractor(
        'pattern', lambda observation: [repr(pattern) for pattern in
                                         observation.part.matching_patterns]),
    SetValuedFeatureExtractor(
        'pattern+conn_parse_path',
        lambda observation: [ # Use quotes to work around stupid ws issue
            '%s / "%s"' % (pattern, ArgumentLabelerModel.get_connective_parse_path(
                           observation))
            for pattern in observation.part.matching_patterns]),
    FeatureExtractor('conn_rel_pos',
                     ArgumentLabelerModel.get_connective_relative_position),
    FeatureExtractor('is_alnum', lambda observation: (
                                    observation.observation.lemma.isalnum()),
                     FeatureExtractor.FeatureTypes.Numerical)
]


class ArgumentLabelerStage(Stage):
    def __init__(self, name, training_algorithm=None, training_params={}):
        if training_algorithm is None:
            training_algorithm = FLAGS.arg_label_training_alg
        super(ArgumentLabelerStage, self).__init__(
            name, [ArgumentLabelerModel(training_algorithm, training_params)])

    def _extract_parts(self, sentence, is_train):
        if is_train:
            # Filter to possible causations for which we can actually extract
            # the correct labels, i.e., gold-standard causations.
            return [possible_causation for possible_causation
                    in sentence.possible_causations
                    if possible_causation.true_causation_instance]
        else:
            return sentence.possible_causations

    def _make_evaluator(self):
        return IAAEvaluator(False, False, FLAGS.arg_label_log_differences,
                            True, True)
