from copy import copy
from gflags import FLAGS

from iaa import CausalityMetrics
from pipeline import Stage

class PossibleCausation(object):
    def __init__(self, matching_pattern, connective_tokens,
                 true_causation_instance=None, cause_tokens=None,
                 effect_tokens=None):
        # There must be at least 1 connective token, or it's not a valid
        # potential instance anyway.
        self.sentence = connective_tokens[0].parent_sentence
        self.matching_pattern = matching_pattern
        self.connective_tokens = connective_tokens
        self.true_causation_instance = true_causation_instance
        # Cause/effect spans are filled in by the second stage.
        self.cause_tokens = cause_tokens
        self.effect_tokens = effect_tokens
        # TODO: Add spans of plausible ranges for argument spans

class IAAEvaluatedStage(Stage):
    def __init__(self, name, models, compare_degrees, compare_types,
                 log_differences):
        super(IAAEvaluatedStage, self).__init__(name, models)
        self.log_differences = log_differences
        self.compare_degrees = compare_degrees
        self.compare_types = compare_types
        # Used during evaluation
        self._predicted_sentences = None
        self._gold_sentences = None

    def _begin_evaluation(self):
        self._predicted_sentences = []
        self._gold_sentences = []

    def _evaluate(self, sentences, original_sentences):
        # TODO: replace storing everything with incrementally building
        # CausalityMetrics.
        self._gold_sentences.extend(original_sentences)
        self._predicted_sentences.extend(sentences)

    _PERMISSIVE_KEY = 'Allowing partial matches'
    _STRICT_KEY = 'Not allowing partial matches'

    def _complete_evaluation(self):
        with_partial = CausalityMetrics(
            self._gold_sentences, self._predicted_sentences, True,
            self.log_differences, compare_degrees=self.compare_degrees,
            compare_types=self.compare_types)
        without_partial = CausalityMetrics(
            self._gold_sentences, self._predicted_sentences, False,
            self.log_differences, compare_degrees=self.compare_degrees,
            compare_types=self.compare_types)

        # TODO: actually log differences here

        self._gold_sentences = []
        self._predicted_sentences = []
        return {self._PERMISSIVE_KEY: with_partial,
                self._STRICT_KEY: without_partial}

    @staticmethod
    def aggregate_eval_results(results_list):
        permissive = CausalityMetrics.aggregate(
            [result_dict[IAAEvaluatedStage._PERMISSIVE_KEY]
             for result_dict in results_list])
        strict = CausalityMetrics.aggregate(
            [result_dict[IAAEvaluatedStage._STRICT_KEY]
             for result_dict in results_list])
        return {IAAEvaluatedStage._PERMISSIVE_KEY: permissive,
                IAAEvaluatedStage._STRICT_KEY: strict}
