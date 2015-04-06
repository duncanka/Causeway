from __future__ import absolute_import
from nltk.tree import ImmutableParentedTree
from numpy import zeros
import unittest

from util.nltk import nltk_tree_to_graph
from tests.util.scipy import ScipyTestCase


class GraphConversionTest(ScipyTestCase):
    def compare_graph_to_correct(self, graph, correct_index_pairs):
        correct_graph = zeros(graph.shape, dtype=bool)
        for start, end in correct_index_pairs:
            correct_graph[start, end] = True
        self.assertArraysEqual(correct_graph, graph.toarray())

    def testSmallTree(self):
        tree = ImmutableParentedTree.fromstring(
            "(ROOT (S (NP (PRP I)) (VP (VB like) (NP (NN fish)))))")
        graph = nltk_tree_to_graph(tree)
        self.assertEqual((8, 8), graph.shape)
        self.compare_graph_to_correct(
            graph, [(0, 1), (1, 2), (2, 3), (1, 4), (4, 5), (4, 6), (6, 7)])

if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
