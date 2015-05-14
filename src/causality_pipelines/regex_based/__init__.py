from copy import copy
from gflags import FLAGS

from data import ParsedSentence
from iaa import CausalityMetrics
from pipeline import Stage, Evaluator
from util import listify, print_indented

class PossibleCausation(object):
    def __init__(self, matching_patterns, connective_tokens,
                 true_causation_instance=None, cause_tokens=None,
                 effect_tokens=None):
        # There must be at least 1 connective token, or it's not a valid
        # potential instance anyway.
        self.sentence = connective_tokens[0].parent_sentence
        self.matching_patterns = listify(matching_patterns)
        self.connective = connective_tokens
        self.true_causation_instance = true_causation_instance
        # Cause/effect spans are filled in by the second stage.
        self.cause = cause_tokens
        self.effect = effect_tokens
        # TODO: Add spans of plausible ranges for argument spans


class IAAEvaluator(Evaluator):
    def __init__(self, compare_degrees, compare_types,
                 log_differences, eval_possible_causations):
        self._with_partial_metrics = CausalityMetrics(
            [], [], True, log_differences, None, compare_degrees,
            compare_types)
        self._without_partial_metrics = CausalityMetrics(
            [], [], False, log_differences, None, compare_degrees,
            compare_types)
        self.eval_possible_causations = eval_possible_causations
        self.compare_degrees = compare_degrees
        self.compare_types = compare_types
        self.log_differences = log_differences

    def evaluate(self, sentences, original_sentences):
        if self.eval_possible_causations:
            # If we're evaluating using possible causations, we need to fake
            # having actual CausationInstances attached to the sentences for IAA
            # code to be happy.
            original_causations = [] # original causations by sentence
            for sentence in sentences:
                original_causations.append(sentence.causation_instances)
                sentence.causation_instances = []
                for possible_causation in sentence.possible_causations:
                    sentence.add_causation_instance(
                        connective=possible_causation.connective,
                        cause=possible_causation.cause,
                        effect=possible_causation.effect)

        with_partial = CausalityMetrics(
            original_sentences, sentences, True, self.log_differences,
            compare_degrees=self.compare_degrees,
            compare_types=self.compare_types)
        without_partial = CausalityMetrics(
            original_sentences, sentences, False, self.log_differences,
            compare_degrees=self.compare_degrees,
            compare_types=self.compare_types)

        if self.log_differences:
            # Prevent clogging the output with connective differences
            without_partial.gold_only_instances = []
            without_partial.predicted_only_instances = []
            with_partial.gold_only_instances = []
            with_partial.predicted_only_instances = []
            print 'Differences not allowing partial matches:'
            without_partial.pp(log_stats=False, log_confusion=False,
                               log_differences=True, indent=1)
            print
            print 'Differences allowing partial matches:'
            with_partial.pp(log_stats=False, log_confusion=False,
                            log_differences=True, indent=1)

        self._with_partial_metrics += with_partial
        self._without_partial_metrics += without_partial

        if self.eval_possible_causations:
            for sentence, causations in zip(sentences, original_causations):
                sentence.causation_instances = causations

    _PERMISSIVE_KEY = 'Allowing partial matches'
    _STRICT_KEY = 'Not allowing partial matches'

    def complete_evaluation(self):
        result = {self._PERMISSIVE_KEY: self._with_partial_metrics,
                  self._STRICT_KEY: self._without_partial_metrics}
        self._with_partial_metrics = None
        self._without_partial_metrics = None
        return result

    def aggregate_results(self, results_list):
        permissive = CausalityMetrics.aggregate(
            [result_dict[IAAEvaluator._PERMISSIVE_KEY]
             for result_dict in results_list])
        strict = CausalityMetrics.aggregate(
            [result_dict[IAAEvaluator._STRICT_KEY]
             for result_dict in results_list])
        return {IAAEvaluator._PERMISSIVE_KEY: permissive,
                IAAEvaluator._STRICT_KEY: strict}
