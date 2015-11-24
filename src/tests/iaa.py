from __future__ import absolute_import

from copy import deepcopy, copy
import gflags
import os
import unittest

from data.readers import CausalityStandoffReader
from iaa import CausalityMetrics
from tests import get_sentences_from_file
from util.metrics import ClassificationMetrics, AccuracyMetrics

class CausalityMetricsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        gflags.FLAGS.metrics_log_raw_counts = True
    
    @staticmethod
    def _get_sentences_with_swapped_args(sentences):
        swapped_sentences = []
        for sentence in sentences:
            swapped_sentence = deepcopy(sentence)
            for instance in swapped_sentence.causation_instances:
                instance.cause, instance.effect = (
                    instance.effect, instance.cause)
            swapped_sentences.append(swapped_sentence)
        return swapped_sentences

    def _test_metrics(
        self, metrics, correct_connective_metrics,
        correct_cause_span_metrics, correct_effect_span_metrics,
        correct_cause_head_metrics, correct_effect_head_metrics,
        correct_cause_jaccard, correct_effect_jaccard):

        self.assertEqual(correct_connective_metrics,
                         metrics.connective_metrics)

        self.assertEqual(correct_cause_span_metrics,
                         metrics.cause_span_metrics)
        self.assertEqual(correct_effect_span_metrics,
                         metrics.effect_span_metrics)
        self.assertEqual(correct_cause_head_metrics,
                         metrics.cause_head_metrics)
        self.assertEqual(correct_effect_head_metrics,
                         metrics.effect_head_metrics)

        self.assertAlmostEqual(correct_cause_jaccard, metrics.cause_jaccard)
        self.assertAlmostEqual(correct_effect_jaccard, metrics.effect_jaccard)

        # TODO: verify type and degree matrices (may require editing data)

    def setUp(self):
        # Unmodified file contains 7 instances.
        self.sentences = get_sentences_from_file(CausalityStandoffReader,
                                                 'IAATest', 'iaa_test.ann')
        # We have 5 unmodified connectives; 1 connective with an added fragment
        # (still qualifies for partial overlap, so 1 FN + 1 FP if matching
        # without partial overlap); 1 missing connectives (FN); and 1 added
        # connective (FP). One of the unmodified connectives has only 1 arg.
        #
        # We also have 1 cause adjusted to partially overlap; 1 cause deleted;
        # and 1 cause changed to a completely different span.
        self.modified_sentences = get_sentences_from_file(
            CausalityStandoffReader, 'IAATest', 'iaa_test_modified.ann')

    def test_same_annotations_metrics(self):
        correct_connective_metrics = ClassificationMetrics(7, 0, 0)
        correct_arg_metrics = AccuracyMetrics(7, 0)
        swapped = self._get_sentences_with_swapped_args(self.sentences)
        for sentences in [self.sentences, swapped]:
            metrics = CausalityMetrics(sentences, sentences, False)
            self._test_metrics(metrics, correct_connective_metrics,
                               *([correct_arg_metrics] * 4 + [1.0] * 2))

    def test_modified_annotations_metrics(self):
        # For non-partial matching, the partial overlap counts as 1 FP + 1 FN.
        correct_connective_metrics = ClassificationMetrics(5, 2, 2)
        correct_cause_span_metrics = AccuracyMetrics(3, 2)
        correct_cause_head_metrics = correct_cause_span_metrics
        correct_effect_span_metrics = AccuracyMetrics(4, 1)
        correct_effect_head_metrics = AccuracyMetrics(5, 0)
        correct_cause_jaccard = 0.6
        correct_effect_jaccard = 33 / 35.

        metrics = CausalityMetrics(self.sentences, self.modified_sentences,
                                   False)
        self._test_metrics(
            metrics, correct_connective_metrics, correct_cause_span_metrics,
            correct_effect_span_metrics, correct_cause_head_metrics,
            correct_effect_head_metrics, correct_cause_jaccard,
            correct_effect_jaccard)

        swapped_sentences, swapped_modified = [
            self._get_sentences_with_swapped_args(s) for s
            in self.sentences, self.modified_sentences]
        swapped_metrics = CausalityMetrics(swapped_sentences, swapped_modified,
                                           False)
        # Swap all the correct arguments
        self._test_metrics(
            swapped_metrics, correct_connective_metrics,
            correct_effect_span_metrics, correct_cause_span_metrics,
            correct_effect_head_metrics, correct_cause_head_metrics,
            correct_effect_jaccard, correct_cause_jaccard)

    def test_add_metrics(self):
        metrics = CausalityMetrics(self.sentences,
                                   self.modified_sentences, False)
        modified_metrics = copy(metrics)
        modified_metrics.cause_jaccard = 0.3
        modified_metrics.effect_jaccard = 1.0
        summed_metrics = metrics + modified_metrics

        correct_connective_metrics = ClassificationMetrics(10, 4, 4)
        correct_cause_span_metrics = AccuracyMetrics(6, 4)
        correct_cause_head_metrics = correct_cause_span_metrics
        correct_effect_span_metrics = AccuracyMetrics(8, 2)
        correct_effect_head_metrics = AccuracyMetrics(10, 0)
        correct_cause_jaccard = 0.45
        correct_effect_jaccard = 34 / 35.
        self._test_metrics(
            summed_metrics, correct_connective_metrics,
            correct_cause_span_metrics, correct_effect_span_metrics,
            correct_cause_head_metrics, correct_effect_head_metrics,
            correct_cause_jaccard, correct_effect_jaccard)

    def test_aggregate_metrics(self):
        metrics = CausalityMetrics(self.sentences,
                                   self.modified_sentences, False)
        aggregated = CausalityMetrics.aggregate([metrics] * 3)
        self._test_metrics(
            metrics, aggregated.connective_metrics,
            aggregated.cause_span_metrics, aggregated.effect_span_metrics,
            aggregated.cause_head_metrics, aggregated.effect_head_metrics,
            aggregated.cause_jaccard, aggregated.effect_jaccard)

        self_metrics = CausalityMetrics(self.sentences, self.sentences, False)
        aggregated = CausalityMetrics.aggregate([metrics, self_metrics])

        correct_connective_metrics = ClassificationMetrics(6, 1, 1)
        correct_cause_span_metrics = AccuracyMetrics(5, 1)
        correct_cause_span_metrics.accuracy = 0.8
        correct_cause_head_metrics = correct_cause_span_metrics
        correct_effect_span_metrics = AccuracyMetrics(5.5, 0.5)
        correct_effect_span_metrics.accuracy = 0.9
        correct_effect_head_metrics = AccuracyMetrics(6, 0)
        correct_effect_head_metrics.accuracy = 1.0
        correct_cause_jaccard = (1.0 + 0.6) / 2.0
        correct_effect_jaccard = 34 / 35.
        self._test_metrics(
            aggregated, correct_connective_metrics,
            correct_cause_span_metrics, correct_effect_span_metrics,
            correct_cause_head_metrics, correct_effect_head_metrics,
            correct_cause_jaccard, correct_effect_jaccard)

if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
