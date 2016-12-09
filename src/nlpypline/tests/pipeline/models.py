from collections import defaultdict
from mock import call, MagicMock
import numpy as np
from sklearn.utils.mocking import CheckingClassifier
from scipy.sparse import lil_matrix, vstack
import unittest

from nlpypline.pipeline.featurization import FeatureExtractor
from nlpypline.pipeline.models import (ClassBalancingClassifierWrapper,
                                       ClassifierModel)
from nlpypline.pipeline.models.structured import Semiring, ViterbiDecoder


class SmallClassBalancingTest(unittest.TestCase):
    ZERO_ROW_VALS = [7, 8, 9]

    def setUp(self):
        self.data = lil_matrix(
            [[1, 2, 3],
             [4, 5, 6],
             SmallClassBalancingTest.ZERO_ROW_VALS,
             [10, 11, 12]], dtype='float')
        self.labels = np.array([1, 1, 0, 1])

    def test_unchanged_for_low_ratios(self):
        for ratio in [0.5, 1.0]:
            data, labels = ClassBalancingClassifierWrapper.rebalance(
                self.data, self.labels, ratio)
            self.assertEqual(data.shape, self.data.shape)
            self.assertEqual(labels.shape, self.labels.shape)

            self.assertTrue((data == self.data).toarray().all())
            self.assertTrue((labels == self.labels).all())

    @staticmethod
    def as_list_of_lists(sparse_matrix):
        return [[x for x in y] for y in sparse_matrix.toarray()]

    def _test_for_count(self, data, labels, count):
        # NOTE: This method relies on the values in the data matrix being
        # unique, or at least having consistent labels for row values.
        original_labels_by_row_val = {}
        for row, label in zip(self.as_list_of_lists(self.data), labels):
            original_labels_by_row_val[tuple(row)] = label

        counts_by_row_val = defaultdict(int)
        for row, label in zip(self.as_list_of_lists(data), labels):
            row = tuple(row)
            original_label = original_labels_by_row_val[row]
            self.assertEqual(original_label, label,
                             "Row %s with label %d has gotten label %d"
                                % (list(row), original_label, label))
            counts_by_row_val[row] += 1

        self.assertEqual(counts_by_row_val[(1.,2.,3.)], 1)
        self.assertEqual(counts_by_row_val[(4.,5.,6.)], 1)
        self.assertEqual(counts_by_row_val[tuple(self.ZERO_ROW_VALS)], count)
        self.assertEqual(counts_by_row_val[(10.,11.,12.)], 1)

    def test_unlimited_balancing(self):
        data, labels = ClassBalancingClassifierWrapper.rebalance(
            self.data, self.labels)
        self._test_for_count(data, labels, 3)

    def test_normal_balancing(self):
        for ratio in [2.0, 3.0]:
            data, labels = ClassBalancingClassifierWrapper.rebalance(
                self.data, self.labels, ratio)
            self._test_for_count(data, labels, int(ratio))

    def test_fractional_balancing(self):
        for ratio in [1.7, 2.2]:
            data, labels = ClassBalancingClassifierWrapper.rebalance(
                self.data, self.labels, ratio)
            self._test_for_count(data, labels, int(ratio))

    def test_balancing_with_uneven_multiples(self):
        # Add two new 0 rows, and 5 new 1 rows. Now the existing ratio of
        # labels is 3:8. This will force rebalancing to have 2 leftover rows
        # when it tries doing full replications. This makes this a good test
        # case for the regression of making the difference negative.
        self.data = vstack(
            [self.data, self.ZERO_ROW_VALS, self.ZERO_ROW_VALS]
                + ([[13, 13, 13]] * 5),
            format='lil')
        self.labels = np.append(self.labels, [0, 0] + [1] * 5)
        data, labels = ClassBalancingClassifierWrapper.rebalance(
            self.data, self.labels, 10)
        # Rebalancing should have added 5 copies of the zero row, for a total of 8.
        self._test_for_count(data, labels, 8)


class ViterbiTest(unittest.TestCase):
    def setUp(self):
        self.decoder = ViterbiDecoder(['S1', 'S2'])

    def test_noop_transitions(self):
        scores = np.array([[0.1, 0.3, 0.1],
                           [0.2, 0.2, 0.3]])
        transitions = np.ones((2, 2))
        best_score, best_path = self.decoder.run_viterbi(scores, transitions)
        self.assertEqual(best_path, ['S2', 'S1', 'S2'])
        self.assertEqual(0.018, best_score)

    def test_real_transitions(self):
        scores = np.array([[0.1, 0.3, 0.1],
                           [0.2, 0.2, 0.3]])
        transitions = np.array([[0.1, 0.2],
                                [0.3, 0.4]])
        best_score, best_path = self.decoder.run_viterbi(scores, transitions)
        self.assertEqual(best_path, ['S2', 'S2', 'S2'])
        self.assertEqual(0.2 * 0.4 * 0.2 * 0.4 * 0.3, best_score)

    def test_sequence_specific_transitions(self):
        scores = np.array([[0.1, 0.3, 0.1],
                           [0.2, 0.2, 0.3]])
        transitions = np.array([[[0.1, 0.9], # 2 matrices for the 2 transitions
                                 [0.3, 0.4]],
                                [[0.1, 0.2],
                                 [0.3, 0.4]]])
        best_score, best_path = self.decoder.run_viterbi(scores, transitions)
        self.assertEqual(best_path, ['S1', 'S2', 'S2'])
        self.assertEqual(0.1 * 0.9 * 0.2 * 0.4 * 0.3, best_score)


class ViterbiMaxPlusTest(unittest.TestCase):
    def setUp(self):
        self.decoder = ViterbiDecoder(['S1', 'S2'], Semiring.MAX_PLUS)

    def test_noop_transitions(self):
        scores = np.array([[0.1, 0.3, 0.1],
                           [0.2, 0.2, 0.3]])
        transitions = np.zeros((2, 2))
        best_score, best_path = self.decoder.run_viterbi(scores, transitions)
        self.assertEqual(best_path, ['S2', 'S1', 'S2'])
        self.assertEqual(0.8, best_score)

    def test_real_transitions(self):
        scores = np.array([[0.1, 0.3, 0.1],
                           [0.2, 0.2, 0.3]])
        transitions = np.array([[0.1, 0.2],
                                [0.3, 0.5]])
        best_score, best_path = self.decoder.run_viterbi(scores, transitions)
        self.assertEqual(best_path, ['S2', 'S2', 'S2'])
        self.assertEqual(1.7, best_score)


class ClassifierTest(unittest.TestCase):
    def testClassifierModelCalls(self):
        I1, I2, I3, I4 = 100, 200, 300, 400
        LABELS = [3, 3]

        test1 = MagicMock(side_effect=lambda i: i)
        test2 = MagicMock(side_effect=lambda i: i)
        Num = FeatureExtractor.FeatureTypes.Numerical
        class TestClassifierModel(ClassifierModel):
            all_feature_extractors = [FeatureExtractor("test1", test1, Num),
                                      FeatureExtractor("test2", test2, Num)]

            @classmethod
            def _get_feature_extractor_groups(klass):
                return [klass.all_feature_extractors]


        c = CheckingClassifier()
        m = TestClassifierModel(classifier=c,
                                selected_features=["test1", "test2"])
        c.fit = MagicMock()
        c.predict = MagicMock(return_value=np.array([1, 0]))
        m._get_gold_labels = MagicMock(return_value=LABELS)

        m.train([I1, I2])

        test1.assert_has_calls([call(I1), call(I2)])
        test2.assert_has_calls([call(I1), call(I2)])
 
        self.assertEqual(1, len(c.fit.call_args_list))
        self.assertEqual(2, len(c.fit.call_args[0]))
        np.testing.assert_array_equal(np.array(([[I1, I1], [I2, I2]])),
                                      c.fit.call_args[0][0].toarray())
        self.assertEqual(LABELS, c.fit.call_args[0][1])
        m._get_gold_labels.assert_called_once_with([I1, I2])

        test1.reset_mock()
        test2.reset_mock()

        m.test([I3, I4])

        test1.assert_has_calls([call(I3), call(I4)])
        test2.assert_has_calls([call(I3), call(I4)])

        self.assertEqual(1, len(c.predict.call_args_list))
        self.assertEqual(1, len(c.predict.call_args[0]))
        np.testing.assert_array_equal(np.array(([[I3, I3], [I4, I4]])),
                                      c.predict.call_args[0][0].toarray())
