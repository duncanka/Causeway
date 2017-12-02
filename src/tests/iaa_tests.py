from __future__ import absolute_import

from copy import deepcopy
import gflags
import unittest

from causeway.because_data import CausalityStandoffReader
from causeway.because_data.iaa import CausalityMetrics
from nlpypline.tests import get_sentences_from_file
from nlpypline.util.metrics import ClassificationMetrics, f1

gflags.FLAGS([]) # Prevent UnparsedFlagAccessError


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
        correct_cause_jaccard, correct_effect_jaccard):

        self.assertEqual(correct_connective_metrics,
                         metrics.connective_metrics)

        self.assertEqual(correct_cause_span_metrics,
                         metrics.cause_metrics.span_metrics)
        self.assertEqual(correct_effect_span_metrics,
                         metrics.effect_metrics.span_metrics)

        self.assertAlmostEqual(correct_cause_jaccard,
                               metrics.cause_metrics.jaccard)
        self.assertAlmostEqual(correct_effect_jaccard,
                               metrics.effect_metrics.jaccard)

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
        correct_cause_metrics = ClassificationMetrics(6, 0, 0)
        correct_effect_metrics = correct_connective_metrics
        swapped = self._get_sentences_with_swapped_args(self.sentences)
        correct_arg_metrics = [correct_cause_metrics, correct_effect_metrics]
        for sentences, arg_metrics in zip(
            [self.sentences, swapped], [correct_arg_metrics, 
                                        list(reversed(correct_arg_metrics))]):
            metrics = CausalityMetrics(sentences, sentences, False)
            self._test_metrics(metrics, correct_connective_metrics,
                               *(arg_metrics + [1.0] * 2))

    def test_modified_annotations_metrics(self):
        # For non-partial matching, the partial overlap counts as 1 FP + 1 FN.
        correct_connective_metrics = ClassificationMetrics(5, 2, 2)
        correct_cause_span_metrics = ClassificationMetrics(2, 3, 4)
        correct_effect_span_metrics = ClassificationMetrics(4, 3, 3)
        correct_cause_jaccard = 0.6
        correct_effect_jaccard = 33 / 35.

        metrics = CausalityMetrics(self.sentences, self.modified_sentences,
                                   False)
        self._test_metrics(
            metrics, correct_connective_metrics, correct_cause_span_metrics,
            correct_effect_span_metrics, correct_cause_jaccard,
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
            correct_effect_jaccard, correct_cause_jaccard)

    def test_add_metrics(self):
        metrics = CausalityMetrics(self.sentences,
                                   self.modified_sentences, False)
        modified_metrics = deepcopy(metrics)
        modified_metrics.cause_metrics.jaccard = 0.3
        modified_metrics.effect_metrics.jaccard = 1.0
        summed_metrics = metrics + modified_metrics

        correct_connective_metrics = ClassificationMetrics(10, 4, 4)
        correct_cause_span_metrics = ClassificationMetrics(4, 6, 8)
        correct_effect_span_metrics = ClassificationMetrics(8, 6, 6)
        correct_cause_jaccard = 0.45
        correct_effect_jaccard = 34 / 35.
        self._test_metrics(
            summed_metrics, correct_connective_metrics,
            correct_cause_span_metrics, correct_effect_span_metrics,
            correct_cause_jaccard, correct_effect_jaccard)

    def test_aggregate_metrics(self):
        metrics = CausalityMetrics(self.sentences,
                                   self.modified_sentences, False)
        aggregated = CausalityMetrics.aggregate([metrics] * 3)
        self._test_metrics(
            aggregated, metrics.connective_metrics,
            metrics.cause_metrics.span_metrics,
            metrics.effect_metrics.span_metrics,
            metrics.cause_metrics.jaccard, metrics.effect_metrics.jaccard)

        self_metrics = CausalityMetrics(self.sentences, self.sentences, False)
        aggregated = CausalityMetrics.aggregate([metrics, self_metrics])

        correct_connective_metrics = ClassificationMetrics(6, 1, 1)

        correct_cause_span_metrics = ClassificationMetrics(4, 1.5, 2)
        correct_cause_span_metrics._precision = (1 + 2/5.) / 2
        correct_cause_span_metrics._recall = (1 + 2/6.) / 2
        correct_cause_span_metrics._f1 = (1 + f1(2/5., 2/6.)) / 2
        correct_cause_jaccard = (1.0 + 0.6) / 2.0

        correct_effect_span_metrics = ClassificationMetrics(5.5, 1.5, 1.5)
        effect_p_r_f1 = (4/7. + 1) / 2
        correct_effect_span_metrics._precision = effect_p_r_f1
        correct_effect_span_metrics._recall = effect_p_r_f1
        correct_effect_span_metrics._f1 = effect_p_r_f1
        correct_effect_jaccard = 34 / 35.

        self._test_metrics(
            aggregated, correct_connective_metrics,
            correct_cause_span_metrics, correct_effect_span_metrics,
            correct_cause_jaccard, correct_effect_jaccard)

if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
