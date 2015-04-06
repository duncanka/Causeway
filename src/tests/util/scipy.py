from __future__ import absolute_import
import numpy as np
from scipy.sparse import lil_matrix
import unittest

from util.scipy import dreyfus_wagner, longest_path_in_tree, \
    UnconnectedNodesError

class ScipyTestCase(unittest.TestCase):
    def assertArraysEqual(self, array1, array2):
        self.assertEqual(
            type(array1), np.ndarray,
            'array1 is not an array (actual type: %s)' % type(array1))
        self.assertEqual(
            type(array2), np.ndarray,
            'array2 is not an array (actual type: %s)' % type(array2))

        self.assertEqual(array1.shape, array2.shape,
                         'Array shapes do not match (%s vs %s)'
                         % (array1.shape, array2.shape))

        if not array1.dtype.isbuiltin or not array2.dtype.isbuiltin:
            self.assertEqual(
                array1.dtype, array2.dtype,
                'Incompatible dtypes: %s vs %s' % (array1.dtype, array2.dtype))

        comparison = (array1 == array2)
        if comparison.all():
            return
        else:
            num_differing = comparison.size - np.count_nonzero(comparison)
            msg = ('Arrays differ at %d locations\n%s\n\nvs.\n\n%s'
                   % (num_differing, array1, array2))
            self.fail(msg)


class DreyfusWagnerTestCase(ScipyTestCase):
    def _test_graph(self, terminals, correct_nodes, correct_graph):
        steiner_nodes, steiner_graph = dreyfus_wagner(
            self.graph, terminals, directed=False)
        self.assertEqual(set(steiner_nodes), set(correct_nodes))
        self.assertArraysEqual(steiner_graph.toarray(),
                               correct_graph.toarray())


class SmallGraphDreyfusWagnerTreeTest(DreyfusWagnerTestCase):
    def setUp(self):
        # Network topology: One path goes 0 -> 4, one goes 0 -> 1 -> 4,
        # and one goes 0 -> 2 -> 3 -> 4.
        graph = lil_matrix((5,5), dtype='bool')
        for index in [(0, 4), (0, 1), (1, 4), (0, 2), (2, 3), (3, 4)]:
            graph[index] = True
        self.graph = graph

    def test_finds_trivial_path(self):
        correct_graph = lil_matrix((5,5), dtype='bool')
        correct_graph[0, 4] = True
        self._test_graph([0, 4], [], correct_graph)

    def test_finds_solution_with_one_steiner_node(self):
        self.graph[0, 4] = False # Get rid of the direct connection 0 -> 4
        # Now there are two paths to 4: 0 -> 1 -> 4 and 0 -> 2 -> 3 -> 4. It
        # should choose the shorter one.
        correct_graph = lil_matrix((5,5), dtype='bool')
        correct_graph[0, 1] = True
        correct_graph[1, 4] = True
        self._test_graph([0, 4], [1], correct_graph)


class LargerGraphDreyfusWagnerTreeTest(DreyfusWagnerTestCase):
    def setUp(self):
        # Network topology:
        #           3 -> 4
        #           ^
        #           |
        # 0 -> 1 -> 2 -> 6
        #           |    ^
        #           v    |
        #           5 ----

        graph = lil_matrix((7,7), dtype='bool')
        for index in [(0, 1), (1, 2), (2, 3), (3, 4), (2, 6), (2, 5), (5, 6)]:
            graph[index] = True
        self.graph = graph

    def test_finds_6_and_4_path_regardless_of_order(self):
        correct_graph = lil_matrix((7,7), dtype='bool')
        for index in [(2, 6), (2, 3), (3, 4)]:
            correct_graph[index] = True
        self._test_graph([6, 4], [2, 3], correct_graph)
        self._test_graph([4, 6], [2, 3], correct_graph)

    def test_finds_0_6_and_4_tree(self):
        correct_graph = lil_matrix((7,7), dtype='bool')
        for index in [(0, 1), (1, 2), (2, 6), (2, 3), (3, 4)]:
            correct_graph[index] = True
        self._test_graph([0, 4, 6], [1, 2, 3], correct_graph)

    def test_handles_reverse_weighted_edges(self):
        self.graph = lil_matrix(self.graph, dtype='int')
        # Add reverse edge from 6 -> 2 with high cost
        self.graph[6, 2] = 2
        correct_graph = lil_matrix((7,7), dtype='int')
        for index in [(2, 6), (2, 3), (3, 4)]:
            correct_graph[index] = 1
        self._test_graph([6, 4], [2, 3], correct_graph)

        # Now bump up the cost on the original edge. That should make the
        # algorithm prefer the reverse edge.
        self.graph[2, 6] = 3
        correct_graph[2, 6] = 0
        correct_graph[6, 2] = 2
        self._test_graph([6, 4], [2, 3], correct_graph)


class DreyfusWagnerRegressionsTestCase(DreyfusWagnerTestCase):
    def test_finds_lower_cost_path(self):
        graph = lil_matrix((3, 3), dtype='float')
        graph[0, 2] = 1
        graph[1, 2] = 0.85
        graph[0, 1] = 0.8
        self.graph = graph
        correct_graph = lil_matrix((3, 3), dtype='float')
        correct_graph[0, 1] = 0.8
        correct_graph[1, 2] = 0.85
        self._test_graph([0, 1, 2], [], correct_graph)

    def test_handles_unconnected_terminal(self):
        graph = lil_matrix((4, 4), dtype='float')
        self.assertRaises(UnconnectedNodesError, dreyfus_wagner, graph,
                          [0, 1, 2])

    def test_consistent_answers_with_cycle(self):
        # Graph topology: 1 ---> 0
        #                 ^
        #                 |--> 2
        graph = lil_matrix((3, 3), dtype='float')
        for index in [(1, 0), (1, 2)]:
            graph[index] = 1.0
        graph[2, 1] = 1.01
        self.graph = graph

        correct_graph = lil_matrix((3, 3), dtype='float')
        correct_graph[1, 0] = 1
        correct_graph[1, 2] = 1

        self._test_graph([2, 0], [1], correct_graph)
        self._test_graph([0, 2], [1], correct_graph)


class LongestPathTestCase(ScipyTestCase):
    def assertArraysEqualOrReversed(self, array1, array2):
        try:
            # Try reversed first. That way, if both fail, the error message
            # show the original array.
            self.assertArraysEqual(array1, array2[::-1])
        except AssertionError:
            self.assertArraysEqual(array1, array2)

    def test_linear_tree(self):
        graph = lil_matrix((5,5), dtype='bool')
        for i in range(4):
            graph[i, i+1] = True
        longest_path = longest_path_in_tree(graph)
        # Either order for the path is technically OK.
        self.assertArraysEqualOrReversed(longest_path,
                                         np.array([0, 1, 2, 3, 4]))

    def test_complex_tree(self):
        '''
        Network topology:

        0 -> 2 -> 5 -> 8
        |    |
        v    v
        1    4 -> 7 -> 9
        |    |
        v    v
        3    6
        '''
        graph = lil_matrix((10,10), dtype='bool')
        edges = [(0,1), (1,3), (0,2), (2,4), (4,6), (4,7), (7,9), (2,5), (5,8)]
        for source, target in edges:
            graph[source, target] = True
        longest_path = longest_path_in_tree(graph)
        self.assertArraysEqualOrReversed(longest_path,
                                         np.array([3, 1, 0, 2, 4, 7, 9]))

        # Now move node 9 to be under node 8, instead of node 7. The path
        # should change accordingly.
        graph[7, 9] = False
        graph[8, 9] = True
        longest_path = longest_path_in_tree(graph)
        self.assertArraysEqualOrReversed(longest_path,
                                         np.array([3, 1, 0, 2, 5, 8, 9]))

    def test_single_node(self):
        graph = lil_matrix((1, 1), dtype='bool')
        longest_path = longest_path_in_tree(graph)
        self.assertArraysEqual(longest_path, np.array([0]))

    def test_starting_from(self):
        graph = lil_matrix((3, 3), dtype='bool')
        graph[1, 2] = True
        longest_path = longest_path_in_tree(graph, 0)
        self.assertArraysEqual(longest_path, np.array([0]))
        longest_path = longest_path_in_tree(graph, 1)
        self.assertArraysEqualOrReversed(longest_path, np.array([1, 2]))


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
