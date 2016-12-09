from __future__ import absolute_import
import numpy as np
import numpy.ma as ma
from scipy.sparse import (bsr_matrix, coo_matrix, csc_matrix, csr_matrix,
                          lil_matrix)
from sklearn.dummy import DummyClassifier
from sklearn.metrics.classification import accuracy_score
import unittest

from nlpypline.util.scipy import (
    add_rows_and_cols_to_matrix, CycleError, dreyfus_wagner,
    get_incoming_indices, get_outgoing_indices, longest_path_in_tree,
    tarjan_topological_sort, UnconnectedNodesError,
    AutoWeightedVotingClassifier)
from tests import NumpyAwareTestCase


class MatrixInsertionTest(NumpyAwareTestCase):
    TEST_MATRIX = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    def test_append_multiple(self):
        result = add_rows_and_cols_to_matrix(self.TEST_MATRIX, [0, 0, 0, 2])
        correct = np.array([[1, 2, 3, 0, 0], [4, 5, 6, 0, 0], [7, 8, 9, 0, 0],
                            [0] * 5, [0] * 5])
        self.assertArraysEqual(correct, result)

    def test_prepend_multiple(self):
        result = add_rows_and_cols_to_matrix(self.TEST_MATRIX, [2, 0, 0, 0])
        correct = np.array([[0] * 5, [0] * 5, [0, 0, 1, 2, 3], [0, 0, 4, 5, 6],
                           [0, 0, 7, 8, 9]])
        self.assertArraysEqual(correct, result)

    def test_interspersed(self):
        result = add_rows_and_cols_to_matrix(self.TEST_MATRIX, [0, 2, 1, 0])
        correct = np.array([[1, 0, 0, 2, 0, 3], [0] * 6, [0] * 6,
                            [4, 0, 0, 5, 0, 6], [0] * 6, [7, 0, 0, 8, 0, 9]])
        self.assertArraysEqual(correct, result)

class DreyfusWagnerTestCase(NumpyAwareTestCase):
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


class LongestPathTestCase(NumpyAwareTestCase):
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


_ALL_GRAPH_GENERATORS = [
    np.zeros, lambda *args, **kwargs: ma.masked_array(np.zeros(*args,
                                                               **kwargs)),
    bsr_matrix, coo_matrix, csc_matrix, csr_matrix, lil_matrix]

def _generate_graph(graph_generator, size, edges):
    generate_and_convert = [bsr_matrix, coo_matrix, csc_matrix, csr_matrix]
    if graph_generator in generate_and_convert:
        graph = _generate_graph(lil_matrix, size, edges)
        method = getattr(graph, 'to' + graph_generator.__name__.split('_')[0])
        return method()
    else:
        graph = graph_generator(size, dtype=bool)
        for source, target in edges:
            graph[source, target] = True
        return graph

class EdgeFinderTest(unittest.TestCase):
    def _test_on_wikipedia_topological_sort_example(self, edge_fn,
                                                    nodes_with_relatives):
        # Wikipedia topological sort example
        for graph_generator in _ALL_GRAPH_GENERATORS:
            edges = [(0, 3), (1, 3), (1, 4), (2, 4), (2, 7), (3, 5), (3, 6),
                     (4, 6)]
            graph = _generate_graph(graph_generator, (8, 8), edges)

            for node, relatives in nodes_with_relatives:
                retrieved_relatives = edge_fn(graph, node)
                retrieved_relatives = set(retrieved_relatives)
                relatives = set(relatives)
                self.assertEqual(
                    relatives, retrieved_relatives,
                    ('%s != %s (node: %d; graph type: %s)' %
                     (retrieved_relatives, relatives, node,
                      graph.__class__.__name__)))

    def test_incoming_edges(self):
        node_parent_pairs = [(0, []), (1, []), (2, []), (3, [0, 1]),
                             (4, [1, 2]), (5, [3]), (6, [3, 4]), (7, [2])]
        self._test_on_wikipedia_topological_sort_example(
            get_incoming_indices, node_parent_pairs)

    def test_outgoing_edges(self):
        node_child_pairs = [(0, [3]), (1, [3, 4]), (2, [4, 7]), (3, [5, 6]),
                             (4, [6]), (5, []), (6, []), (7, [])]
        self._test_on_wikipedia_topological_sort_example(
            get_outgoing_indices, node_child_pairs)


class TopologicalSortTest(unittest.TestCase):
    def assert_sorted_order(self, sorted_nodes, node_pairs, graph_format):
        for earlier_node, later_node in node_pairs:
            self.assertLess(
                sorted_nodes.index(earlier_node),
                sorted_nodes.index(later_node),
                ('Incorrect sort order: node %d should be before %d (in format'
                 ' %s)' % (earlier_node, later_node, graph_format.__name__)))

    def test_one_level_binary_tree(self):
        for graph_generator in _ALL_GRAPH_GENERATORS:
            edges = [(0, 1), (0, 2)]
            graph = _generate_graph(graph_generator, (3, 3), edges)
            sorted_nodes = tarjan_topological_sort(graph)
            self.assert_sorted_order(sorted_nodes, [(0, 1), (0, 2)],
                                     graph.__class__)

    def test_wikipedia_example(self):
        for graph_generator in _ALL_GRAPH_GENERATORS:
            edges = [(0, 3), (1, 3), (1, 4), (2, 4), (2, 7), (3, 5), (3, 6),
                     (4, 6)]
            graph = _generate_graph(graph_generator, (8, 8), edges)

            sorted_nodes = tarjan_topological_sort(graph)
            ordered_pairs = [
                (0, 3), (0, 5), (0, 6), (1, 3), (1, 5), (1, 6), (1, 4), (2, 4),
                (2, 6), (2, 7), (3, 5), (3, 6), (4, 6)]
            self.assert_sorted_order(sorted_nodes, ordered_pairs,
                                     graph.__class__)

    def test_complains_about_dag(self):
        for graph_generator in _ALL_GRAPH_GENERATORS:
            edges = [[1, 0], [0, 1]]
            graph = _generate_graph(graph_generator, (3, 3), edges)
            self.assertRaises(CycleError,
                              lambda: tarjan_topological_sort(graph))


class AutoWeightedVotingClassifierTest(NumpyAwareTestCase):
    def setUp(self):
        base_c0 = DummyClassifier(strategy='prior')
        base_c0.fit([[], [], []], [0, 0, 1])
        base_c1 = DummyClassifier(strategy='prior')
        base_c1.fit([[], [], []], [0, 1, 1])
        base_classifiers = [('c0', base_c0), ('c1', base_c1)]
        self.classifier = AutoWeightedVotingClassifier(
            base_classifiers, voting='soft', score_fn=accuracy_score)
        self.X = np.array([[0, 0, 0, 2],
                           [0, 0, 0, 4],
                           [1, 1, 1, 3]])
        self.y = np.array([1, 0, 0])

    def test_fit_weights(self):
        self.classifier.fit_weights(self.X, self.y)
        self.assertArraysEqual(np.array([1 / 3., 2 / 3.]),
                               self.classifier.weights)

    def test_predict_before_fit(self):
        probs = self.classifier.predict_proba(self.X)
        expected = np.full((3, 2), 0.5)
        self.assertArraysEqual(expected, probs)

        self.classifier.fit_weights(self.X, self.y)
        expected = (1 / 3. * np.tile([1 / 3., 2 / 3.], [3, 1])
                    + 2 / 3. * np.tile([2 / 3., 1 / 3.], [3, 1]))
        probs = self.classifier.predict_proba(self.X)
        self.assertArraysEqual(expected, probs)

    def test_full_fit(self):
        self.classifier.fit(self.X, self.y)
        expected = np.tile([2 / 3., 1 / 3.], [3, 1])
        probs = self.classifier.predict_proba(self.X)
        self.assertArraysEqual(expected, probs)


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
