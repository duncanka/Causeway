from copy import copy
from gflags import DEFINE_list, DEFINE_string, DEFINE_bool, FLAGS, \
    DuplicateFlagError
import logging

from iaa import CausalityMetrics
from pipeline import Stage
from pipeline.models import CRFModel
from pipeline.feature_extractors import FeatureExtractor
from causality_pipelines.connective_based import PossibleCausation

try:
    DEFINE_list('arg_label_features', ['lemma', 'pos', 'isconnective'],
                'Features for the argument-labeling CRF')
    DEFINE_string('arg_label_model_path', 'arg-labeler-crf.model',
                  'Path to save the argument-labeling CRF model to')
    DEFINE_bool('arg_label_log_differences', False,
                'Whether to log differences between gold and predicted results'
                ' for each round of argument labeling evaluation')
except DuplicateFlagError as e:
    logging.warn('Ignoring redefinition of flag %s' % e.flagname)


class ArgumentLabelerModel(CRFModel):
    CAUSE_LABEL = 'Cause'
    EFFECT_LABEL = 'Effect'
    CONNECTIVE_LABEL = 'Connective'
    
    def __init__(self, training_algorithm, training_params):
        super(ArgumentLabelerModel, self).__init__(
            PossibleCausation, FLAGS.arg_label_model_path,
            self.FEATURE_EXTRACTOR_MAP, FLAGS.arg_label_features,
            training_algorithm, training_params)
    
    def _sequences_for_part(self, part):
        # part for this model is a PossibleCausation.
        observations = part.sentence.tokens
        if part.true_causation_instance:
            labels = ['None'] * len(observations)
            for connective_token in part.connective_tokens:
                labels[connective_token.index] = self.CONNECTIVE_LABEL
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

# Because this is a CRF model operating on sequences of tokens, the input to
# each feature extractor will be a CRFModel.ExtractorPart. The observation
# will be a Token object, and the sequence will be a sequence of Tokens.
FEATURE_EXTRACTORS = [
    FeatureExtractor('lemma', lambda contextful_observation: ( 
                                  contextful_observation.observation.lemma)),
    FeatureExtractor('pos', lambda contextful_observation: (
                                contextful_observation.observation.pos)),
    FeatureExtractor('isconnective',
                     lambda contextful_observation: (
                         contextful_observation.observation in
                         contextful_observation.part.connective_tokens))]
# TODO: These are really sucky features. Add some more.

ArgumentLabelerModel.FEATURE_EXTRACTOR_MAP = {
    extractor.name: extractor for extractor in FEATURE_EXTRACTORS}


class ArgumentLabelerStage(Stage):
    def __init__(self, name, training_algorithm='lbfgs', training_params={}):
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
