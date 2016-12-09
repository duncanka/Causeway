from __future__ import absolute_import
from nltk.tree import ImmutableParentedTree, ParentedTree
from numpy import zeros
import unittest

from nlpypline.util.nltk import nltk_tree_to_graph, get_head, collins_find_heads
from tests import NumpyAwareTestCase


class GraphConversionTest(NumpyAwareTestCase):
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

class HeadFindingTest(NumpyAwareTestCase):
    def _testSmallTree(self, tree):
        heads = collins_find_heads(tree)
        self.assertIs(get_head(heads, tree), tree[0][1][0])
        self.assertIs(get_head(heads, tree[0]), tree[0][1][0])
        self.assertIs(get_head(heads, tree[0][1]), tree[0][1][0])
        self.assertIs(get_head(heads, tree[0][1][1]), tree[0][1][1][0])

    def testSmallImmutableTree(self):
        tree = ImmutableParentedTree.fromstring(
            "(ROOT (S (NP (PRP I)) (VP (VB like) (NP (NN fish))) (. .)))")
        self._testSmallTree(tree)

    def testSmallMutableTree(self):
        tree = ParentedTree.fromstring(
            "(ROOT (S (NP (PRP I)) (VP (VB like) (NP (NN fish))) (. .)))")
        self._testSmallTree(tree)
        
    def testLargerTree(self):
        tree_str = (
            '''
            (ROOT
              (S
                (NP
                  (NP (RB nearly) (DT every) (NN session))
                  (PP (IN since) (NP (NNP November))))
                (VP
                  (VBZ have)
                  (VP
                    (VBN be)
                    (VP
                      (VBN adjourn)
                      (SBAR
                        (IN because)
                        (S
                          (NP (QP (RB as) (JJ few) (IN as) (CD 65))
                              (NNS member))
                          (VP
                            (VBD make)
                            (S (NP (PRP it)) (VP (TO to) (VP (VB work)))))))
                      (, ,)
                      (SBAR
                        (RB even)
                        (IN as)
                        (S
                          (NP
                            (NP (PRP they))
                            (CC and)
                            (NP (DT the) (NNS absentee)))
                          (VP
                            (VBD earn)
                            (NP (NNS salary) (CC and) (NNS benefit))
                            (PP
                              (IN worth)
                              (NP (QP (RB about) ($ $) (CD 120,000))))))))))
                (. .)))
            ''')
        tree = ImmutableParentedTree.fromstring(tree_str)
        heads = collins_find_heads(tree)
        self.assertIs(get_head(heads, tree[0][1]), tree[0][1][1][1][0])
        self.assertIs(get_head(heads, tree[0]), tree[0][1][1][1][0])

if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
