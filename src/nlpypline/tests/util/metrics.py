from math import isnan
import unittest

from nlpypline.util.metrics import ClassificationMetrics, ConfusionMatrix


class ClassificationMetricsTest(unittest.TestCase):
    def setUp(self):
        self.metrics = ClassificationMetrics(tp=15, fp=10, fn=5, tn=1)

    def testBasicMetrics(self):
        self.assertEqual(self.metrics.tp, 15)
        self.assertEqual(self.metrics.fp, 10)
        self.assertEqual(self.metrics.fn, 5)
        self.assertEqual(self.metrics.tn, 1)

    def testDerivedMetrics(self):
        self.assertEqual(self.metrics.precision, 0.6)
        self.assertEqual(self.metrics.recall, 0.75)
        self.assertAlmostEqual(self.metrics.f1, 2 / 3.0)

    def testMetricsWithZeroes(self):
        self.metrics.tp = 0
        self.assertEqual(0, self.metrics.recall)
        self.assertEqual(0, self.metrics.precision)

        self.metrics.fn = 0
        self.assertEqual(0, self.metrics.recall)

        self.metrics.fn = 5
        self.metrics.fp = 0
        self.assertEqual(0, self.metrics.precision)
        self.assertEqual(0, self.metrics.f1)


class ConfusionMatrixTest(unittest.TestCase):
    def setUp(self):
        # Example from:
        # http://alias-i.com/lingpipe/docs/api/com/aliasi/classify/ConfusionMatrix.html
        shared = ['C'] * 9 + ['S'] * 5 + ['P'] * 4
        response = shared + ['S'] * 3 + ['C'] * 3 + ['P', 'C', 'S']
        reference = shared + ['C'] * 3 + ['S'] * 3 + ['S', 'P', 'P']
        self.matrix = ConfusionMatrix(reference, response)

    def testF1s(self):
        self.assertAlmostEqual(2 / 3., self.matrix.f1_micro())
        self.assertAlmostEqual(170116 / 253989., self.matrix.f1_macro())

if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
