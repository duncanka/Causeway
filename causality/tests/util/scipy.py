from __future__ import absolute_import
from scipy.sparse import lil_matrix
from scipy.sparse.csgraph import shortest_path
import unittest

from util.scipy import dreyfus_wagner

class DreyfusWagnerTestCase(unittest.TestCase):
    def _test_graph(self, terminals, correct_nodes, correct_graph):
        path_costs, path_predecessors = shortest_path(
            self.graph, unweighted=True, return_predecessors=True,
            directed=False)
        steiner_nodes, steiner_graph = dreyfus_wagner(
            self.graph, terminals, path_costs, path_predecessors)
        self.assertEqual(set(steiner_nodes), set(correct_nodes))
        self.assertTrue((steiner_graph == correct_graph).toarray().all())


class SmallGraphDreyfusWagnerTreeTest(DreyfusWagnerTestCase):
    def setUp(self):
        # Network topology: One path goes 0 -> 4, one goes 0 -> 1 -> 4,
        # and one goes 0 -> 2 -> 3 -> 4.
        graph = lil_matrix((5,5), dtype='bool')
        graph[0, 4] = True
        graph[0, 1] = True
        graph[1, 4] = True
        graph[0, 2] = True
        graph[2, 3] = True
        graph[3, 4] = True
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
        graph[0, 1] = True
        graph[1, 2] = True
        graph[2, 3] = True
        graph[3, 4] = True
        graph[2, 6] = True
        graph[2, 5] = True
        graph[5, 6] = True
        self.graph = graph

    def test_finds_6_and_4_path_regardless_of_order(self):
        correct_graph = lil_matrix((7,7), dtype='bool')
        correct_graph[2, 6] = True
        correct_graph[2, 3] = True
        correct_graph[3, 4] = True
        self._test_graph([6, 4], [2, 3], correct_graph)
        self._test_graph([4, 6], [2, 3], correct_graph)

    def test_finds_0_6_and_4_tree(self):
        correct_graph = lil_matrix((7,7), dtype='bool')
        correct_graph[0, 1] = True
        correct_graph[1, 2] = True
        correct_graph[2, 6] = True
        correct_graph[2, 3] = True
        correct_graph[3, 4] = True
        self._test_graph([0, 4, 6], [1, 2, 3], correct_graph)

    def test_handles_reverse_weighted_edges(self):
        self.graph = lil_matrix(self.graph, dtype='int')
        # Add reverse edge from 6 -> 2 with high weight
        self.graph[6, 2] = 2
        correct_graph = lil_matrix((7,7), dtype='int')
        correct_graph[2, 6] = 1
        correct_graph[2, 3] = 1
        correct_graph[3, 4] = 1
        self._test_graph([6, 4], [2, 3], correct_graph)

        # Now bump up the weight on the original edge. That should make the
        # algorithm prefer the reverse edge.
        self.graph[2, 6] = 3
        correct_graph[2, 6] = 0
        correct_graph[6, 2] = 2
        self._test_graph([6, 4], [2, 3], correct_graph)


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
