'''
Created on Dec 1, 2014

@author: jesse
'''
from collections import defaultdict
import numpy as np
from scipy.sparse import lil_matrix
import unittest

from pipeline.models import ClassBalancingModelWrapper


class SmallClassBalancingTest(unittest.TestCase):
    def setUp(self):
        self.data = np.array(
            [[1, 2, 3],
             [4, 5, 6],
             [7, 8, 9],
             [10, 11, 12]], dtype='float')
        self.labels = np.array([1, 1, 0, 1])
        
    def test_unchanged_for_low_ratios(self):
        for ratio in [0.5, 1.0]:
            data, labels = ClassBalancingModelWrapper.rebalance(
                self.data, self.labels, ratio)
            self.assertEqual(data.shape, self.data.shape)
            self.assertEqual(labels.shape, self.labels.shape)
            
            self.assertTrue((data == self.data).all())
            self.assertTrue((labels == self.labels).all())
            
    def _test_for_count(self, data, labels, count):
        # NOTE: This method relies on the values in the data matrix being unique.     
        original_labels_by_row_val = {}
        for row, label in zip(self.data.tolist(),
                              self.labels.tolist()):
            original_labels_by_row_val[tuple(row)] = label
        
        counts_by_row_val = defaultdict(int)
        for row, label in zip(data.tolist(), labels.tolist()):
            row = tuple(row)
            original_label = original_labels_by_row_val[row]
            self.assertEqual(original_label, label,
                             "Row %s with label %d has gotten label %d"
                                % (list(row), original_label, label))
            counts_by_row_val[row] += 1
            
        self.assertEqual(counts_by_row_val[(1.,2.,3.)], 1)
        self.assertEqual(counts_by_row_val[(4.,5.,6.)], 1)
        self.assertEqual(counts_by_row_val[(7.,8.,9.)], count)
        self.assertEqual(counts_by_row_val[(10,11,12)], 1)

    def test_unlimited_balancing(self):
        data, labels = ClassBalancingModelWrapper.rebalance(
            self.data, self.labels)
        self._test_for_count(data, labels, 3)
                
    def test_normal_balancing(self):
        for ratio in [2.0, 3.0]:
            data, labels = ClassBalancingModelWrapper.rebalance(
                self.data, self.labels, ratio)
            self._test_for_count(data, labels, int(ratio))
                
    def test_fractional_balancing(self):
        for ratio in [1.7, 2.2]:
            data, labels = ClassBalancingModelWrapper.rebalance(
                self.data, self.labels, ratio)
            self._test_for_count(data, labels, int(ratio))

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()