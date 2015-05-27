from copy import copy
from gflags import DEFINE_bool, FLAGS, DuplicateFlagError
import logging

from data import ParsedSentence
from iaa import CausalityMetrics
from pipeline import Stage, Evaluator
from util import listify, print_indented

class PossibleCausation(object):
    '''
    Designed to mimic actual CausationInstance objects.
    '''
    # TODO: Make these actually descend from CausationInstances.
    def __init__(self, matching_patterns, connective,
                 true_causation_instance=None, cause=None, effect=None):
        # There must be at least 1 connective token, or it's not a valid
        # potential instance anyway.
        self.sentence = connective[0].parent_sentence
        self.matching_patterns = listify(matching_patterns)
        self.connective = connective
        self.true_causation_instance = true_causation_instance
        # Cause/effect spans are filled in by the second stage.
        self.cause = cause
        self.effect = effect
        # TODO: Add spans of plausible ranges for argument spans


class IAAEvaluator(Evaluator):
    # TODO: refactor arguments
    def __init__(self, compare_degrees, compare_types, log_test_instances,
                 compare_args, pairwise_only,
                 causations_property_name='causation_instances'):
        self._with_partial_metrics = CausalityMetrics(
            [], [], True, log_test_instances, None, compare_degrees,
            compare_types, True, True, log_test_instances)
        self._without_partial_metrics = CausalityMetrics(
            [], [], False, log_test_instances, None, compare_degrees,
            compare_types, True, True, log_test_instances)
        self.causations_property_name = causations_property_name
        self.compare_degrees = compare_degrees
        self.compare_types = compare_types
        self.log_test_instances = log_test_instances
        self.compare_args = compare_args
        self.pairwise_only = pairwise_only

    def evaluate(self, sentences, original_sentences):
        with_partial = CausalityMetrics(
            original_sentences, sentences, True, self.log_test_instances,
            compare_types=self.compare_types, compare_args=self.compare_args,
            compare_degrees=self.compare_degrees,
            pairwise_only=self.pairwise_only,
            save_agreements=self.log_test_instances,
            causations_property_name=self.causations_property_name)
        without_partial = CausalityMetrics(
            original_sentences, sentences, False, self.log_test_instances,
            compare_types=self.compare_types, compare_args=self.compare_args,
            compare_degrees=self.compare_degrees,
            pairwise_only=self.pairwise_only,
            save_agreements=self.log_test_instances,
            causations_property_name=self.causations_property_name)

        if self.log_test_instances:
            print 'Differences not allowing partial matches:'
            without_partial.pp(log_stats=False, log_confusion=False,
                               log_differences=True, log_agreements=True,
                               indent=1)
            print
            print 'Differences allowing partial matches:'
            with_partial.pp(log_stats=False, log_confusion=False,
                            log_differences=True, log_agreements=True,
                            indent=1)

        self._with_partial_metrics += with_partial
        self._without_partial_metrics += without_partial

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
