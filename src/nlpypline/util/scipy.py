from __future__ import absolute_import
from itertools import izip
from nltk.metrics.scores import accuracy
import numpy as np
import numpy.ma as ma
from scipy.sparse import lil_matrix
from scipy.sparse.csgraph import shortest_path, breadth_first_order
from sklearn.ensemble.voting_classifier import VotingClassifier
from sklearn.preprocessing.label import LabelEncoder
from sklearn.preprocessing import normalize

from nlpypline.util import pairwise, Enum

def add_rows_and_cols_to_matrix(matrix, indices_to_add):
    '''
    Returns a new matrix with rows and columns inserted in the specified
    locations.
    
    `matrix`: a square Numpy matrix.
    `indices_to_add`: a sequence of length matrix.shape[0] + 1. The value at
                      each index indicates the number of empty rows/columns to
                      insert before that index. 
    '''
    # Add rows
    assert len(matrix.shape) == 2 and matrix.shape[0] == matrix.shape[1], (
        "Can't add equivalent rows/columns in non-square matrix")
    assert len(indices_to_add) == matrix.shape[0] + 1, (
        "Number of indices must be 1 greater than dimension of matrix")
    new_column = np.zeros((matrix.shape[0], 1))
    columns = [matrix[:, [i]] for i in range(matrix.shape[1])]
    matrix_chunks = []
    for column, to_insert_count in zip(columns, indices_to_add):
        if to_insert_count:
            matrix_chunks.extend([new_column] * to_insert_count)
        matrix_chunks.append(column)
    matrix_chunks.extend([new_column] * indices_to_add[-1])
    new_matrix = np.hstack(matrix_chunks)

    new_row = np.zeros((1, new_matrix.shape[1]))
    rows = [new_matrix[i, :] for i in range(new_matrix.shape[0])]
    matrix_chunks = []
    for row, to_insert_count in zip(rows, indices_to_add):
        if to_insert_count:
            matrix_chunks.extend([new_row] * to_insert_count)
        matrix_chunks.append(row)
    matrix_chunks.extend([new_row] * indices_to_add[-1])
    new_matrix = np.vstack(matrix_chunks)

    return new_matrix


def _find_depth(i, predecessors, depths):
    if depths[i] == np.inf:
        pred = predecessors[i]
        if pred != -9999:
            depths[i] = _find_depth(pred, predecessors, depths) + 1
        else:
            depths[i] = np.inf
    return depths[i]

def bfs_shortest_path_costs(graph, start):
    predecessors = breadth_first_order(graph, start, True, True)[1]
    depths = [0] + [np.inf] * (len(predecessors) - 1)
    for i in range(len(predecessors)):
        _find_depth(i, predecessors, depths)
    return depths

# See https://github.com/numpy/numpy/blob/master/numpy/lib/arraysetops.py#L96.
def unique(mat, return_index=False, return_inverse=False, return_counts=False):
    """
    Find the unique elements of a sparse matrix.

    Returns the sorted unique elements of a sparse matrix. There are two
    optional outputs in addition to the unique elements: the indices of the
    input matrix that give the unique values, and the indices of the unique
    matrix that reconstruct the input matrix.

    Parameters
    ----------
    mat : sparse matrix
        Input matrix. This will be converted to the CSR representation
        internally.
    return_index : bool, optional
        If True, also return the indices of `mat` that result in the unique
        array.
    return_inverse : bool, optional
        If True, also return the indices of the unique array that can be used
        to reconstruct `mat`.
    return_counts : bool, optional
        If True, also return the number of times each unique value comes up
        in `mat`.

    Returns
    -------
    unique : ndarray
        The sorted unique values.
    unique_indices : ndarray, optional
        The indices of the first occurrences of the unique values in the
        (flattened) original array. Only provided if `return_index` is True.
    unique_inverse : ndarray, optional
        The indices to reconstruct the (flattened) original array from the
        unique array. Only provided if `return_inverse` is True.

        Note that, because the matrix is sparse, the full array of indices is
        not returned. Instead, an array i is returned such that, given an empty
        sparse matrix m with the same number of columns as there were elements
        in mat, setting m[0, i[0]] = unique[i[1]] will reproduce the original
        matrix.
    unique_counts : ndarray, optional
        The number of times each of the unique values comes up in the
        original array. Only provided if `return_counts` is True.

    See Also
    --------
    numpy.lib.arraysetops.unique : Basis for this function, but only works for
                                   dense matrices/arrays.
    """
    # Convert to CSR because this format has a .data matrix, which is the main
    # thing we need, and which is stored in sorted order of rows -> columns.
    # This means that np.unique returns the indices and inverse in terms of a
    # sensibly linearized matrix. (The nonzero indices are also returned in
    # row -> column order, which is useful for the return_inverse and
    # return_index options.) Also, CSR is fairly memory-efficient and quick to
    # convert to from other formats.
    mat = mat.tocsr()
    size = mat.shape[0] * mat.shape[1] # mat.size just gives nnz

    unique_data = np.unique(mat.data, return_index, return_inverse,
                            return_counts)

    # If there are no zeros, we can just pretend we're operating on a normal
    # dense array. All we have to do then is check whether we need to adapt the
    # inverse return value to our special sparse inverse format.
    if mat.nnz == size:
        if return_inverse:
            inv_index = (2 if return_index else 1)
            inverse = np.vstack((range(size), unique_data[inv_index]))
            unique_data = list(unique_data)
            unique_data[inv_index] = inverse
            unique_data = tuple(unique_data)
        return unique_data

    # OK, there are some zeros.
    # Our lives are simplest if the only thing requested was the unique values.
    if not isinstance(unique_data, tuple):
        # We got here because there are zeros, so we know 0 should be in the
        # list of unique values.
        return np.insert(unique_data, 0, 0.0)

    # If more values were requested, process other return values in the tuple
    # as necessary.
    unique_data = list(reversed(unique_data))
    unique_values = unique_data.pop()
    unique_values = np.insert(unique_values, 0, 0.0)
    ret = (unique_values,)

    # Offset returned indices to account for missing zero entries.
    if return_index or return_inverse:
        if return_index:
            indices = unique_data.pop()
        if return_inverse:
            inverse = unique_data.pop()

            # We're going to use inverse[0] as the array indices at which
            # values in the original matrix reside, and inverse[1] as the
            # indices in the unique array from which to draw those values.
            # We must add 1 to inverse[1] to account for the 0 in the initial
            # position.

            # The indices for the inverse matrix aren't accounting for the
            # presence of a zero value at the start of the list.
            inverse_unique_indices = inverse + 1
            # Initialize positions in original matrix to values' current
            # positions in the inverse array. As we detect 0 values in the
            # original matrix, we'll increase these indices accordingly.
            inverse_orig_pos_indices = np.array(range(len(inverse)))

        first_zero = None
        offset = 0
        mat.sort_indices()
        nonzero = mat.nonzero()

        for i, (row, col) in enumerate(izip(nonzero[0], nonzero[1])):
            offset_i = i + offset
            flattened_index = row * mat.shape[1] + col
            difference = flattened_index - offset_i
            if difference > 0: # We've found one or more zero entries!
                if return_index:
                    indices[np.where(indices >= offset_i)] += difference
                    if first_zero is None:
                        first_zero = i
                        indices = np.insert(indices, 0, first_zero)
                if return_inverse:
                    inverse_orig_pos_indices[
                        np.where(inverse_orig_pos_indices >= offset_i)
                        ] += difference
                offset += difference

        if return_index:
            ret += (indices,)

        if return_inverse:
            inverse = np.vstack((inverse_orig_pos_indices,
                                 inverse_unique_indices))
            ret += (inverse,)

    # Add counts for 0 value.
    if return_counts:
        counts = unique_data.pop()
        counts = np.insert(counts, 0, size - mat.nnz)
        ret += (counts,)

    return ret


class UnconnectedNodesError(Exception):
    pass

def reconstruct_predecessor_path_1d(predecessors, w, v):
    if w == v:
        return np.array([v])

    predecessor = predecessors[v]
    path = [v]
    while predecessor != w:
        if predecessor == -9999:
            raise UnconnectedNodesError(
                'No path between nodes %d and %d' % (w, v))
        path = [predecessor] + path
        predecessor = predecessors[predecessor]
    return np.array([w] + path)

def reconstruct_predecessor_path(predecessors, w, v):
    if len(predecessors.shape) == 1:
        return reconstruct_predecessor_path_1d(predecessors, w, v)

    def helper(w, v):
        predecessor = predecessors[w, v]
        if predecessor == w:
            return [v]
        elif predecessor == -9999:
            raise UnconnectedNodesError(
                'No path between nodes %d and %d' % (w, v))
        else:
            return helper(w, predecessor) + [v]
    return np.array([w] + helper(w, v))

# Steiner tree finding.
def steiner_tree(graph, terminals, *args, **kwargs):
    algorithm = kwargs.pop('algorithm', 'dreyfus-wagner')
    if algorithm == 'dreyfus-wagner':
        return dreyfus_wagner(graph, terminals, *args, **kwargs)
    else:
        raise NotImplementedError

# Based on http://siekiera.mimuw.edu.pl/~paal/dreyfus__wagner_8hpp_source.html.
def dreyfus_wagner(graph, terminals, shortest_path_costs=None,
                   shortest_path_predecessors=None, directed=True,
                   start_terminal_index=0):
    '''
    The Dreyfus-Wagner algorithm assumes a fully connected graph. We simulate
    that by precomputing all shortest paths, or by having them supplied, and
    using the shortest path cost between two nodes as its edge cost. Of course,
    this means we need to do some extra work at the end to reconstruct the
    solution in the original graph.

    If `shortest_path_costs` and `shortest_path_predecessors` are provided,
    `directed` is ignored, as it is only used for shortest-path computation.
    The edge weights in `graph` must match those used to generate these two
    variables, or the behavior may be incorrect.
    '''
    def get_edge_cost(w, v):
        return shortest_path_costs[w, v]

    def best_split(vertex, remaining, subset, index):
        if index == len(terminals):
            complement = remaining ^ subset
            if subset.any() and complement.any():
                cost = (connect_vertex(vertex, subset)
                        + connect_vertex(vertex, complement))
                return (cost, subset)
            else:
                return (np.inf, None)
        else:
            ret1 = best_split(vertex, remaining, subset, index + 1)
            if remaining[index]:
                # Flip the bit at index
                subset = subset.copy() # copy 1st to prevent messing w/ caller
                subset[index] ^= True
                ret2 = best_split(vertex, remaining, subset, index + 1)
                if ret1[0] > ret2[0]:
                    ret1 = ret2
            return ret1

    def split_vertex(vertex, remaining):
        # TODO: optimize away this counting step by passing an arg?
        if np.count_nonzero(remaining) < 2:
            return 0
        # Use the memoized version if possible.
        try:
            return best_splits[(vertex, remaining.tostring())][0]
        except KeyError:
            # Optimization, to avoid checking subset twice.
            remaining_index = np.where(remaining == True)[0][0] + 1
            best = best_split(vertex, remaining, remaining, remaining_index)
            best_splits[(vertex, remaining.tostring())] = best
            return best[0]

    def connect_vertex(vertex, remaining):
        ''' Computes minimal cost of connecting given vertex with all vertices
            in `remaining`. '''
        n_remaining = np.count_nonzero(remaining)
        # Base cases: 0 or 1 remaining nodes
        if n_remaining == 0:
            return 0
        elif n_remaining == 1:
            terminal_index = np.where(remaining == True)[0][0]
            cost = get_edge_cost(vertex, terminals[terminal_index])
            best_candidates[(vertex, remaining.tostring())] = (
                cost, terminals[terminal_index])
            return cost

        # Use the memoized version if possible
        try:
            return best_candidates[(vertex, remaining.tostring())][0]
        except KeyError:
            best_cost = split_vertex(vertex, remaining)
            candidate_vertex = vertex

            finished_terminals = [
                t for (t, t_pos) in terminal_positions.iteritems()
                if not remaining[t_pos]]
            vertices_to_try = non_terminals + finished_terminals

            for vertex_to_try in vertices_to_try:
                cost = split_vertex(vertex_to_try, remaining)
                cost += get_edge_cost(vertex, vertex_to_try)
                if best_cost < 0 or cost < best_cost:
                    best_cost = cost
                    candidate_vertex = vertex_to_try

            remaining = remaining.copy() # Prevent damage to caller state
            for terminal, terminal_position in terminal_positions.iteritems():
                if not remaining[terminal_position]:
                    continue
                remaining[terminal_position] = False
                cost = connect_vertex(terminal, remaining)
                cost += get_edge_cost(vertex, terminal)
                remaining[terminal_position] = True

                if best_cost < 0 or cost < best_cost:
                    best_cost = cost
                    candidate_vertex = terminal

            best_candidates[(vertex, remaining.tostring())] = (
                best_cost, candidate_vertex)
            return best_cost

    def retrieve_solution_connect(vertex, remaining):
        if not remaining.any():
            return
        next_vertex = best_candidates[(vertex, remaining.tostring())][1]

        if vertex == next_vertex: # called wagner directly from dreyfus (?)
            retrieve_solution_split(next_vertex, remaining)
        else:
            steiner_edges.append((vertex, next_vertex))
            # Case 1: Non-terminal or terminal that is not remaining
            if (next_vertex not in terminal_positions) or (
                not remaining[terminal_positions[next_vertex]]):
                retrieve_solution_split(next_vertex, remaining)
            # Case 2: terminal that is remaining
            else:
                terminal_position = terminal_positions[next_vertex]
                # Copy to prevent damage to caller state
                remaining = remaining.copy()
                remaining[terminal_position] ^= 1
                retrieve_solution_connect(next_vertex, remaining)

    def retrieve_solution_split(vertex, remaining):
        # TODO: is this check actually necessary?
        if not remaining.any():
            return
        split = best_splits[(vertex, remaining.tostring())][1]
        if split is None:
            raise UnconnectedNodesError('Could not connect terminal %d' % vertex)
        retrieve_solution_connect(vertex, split)
        retrieve_solution_connect(vertex, remaining ^ split)

    # Initialization

    if shortest_path_predecessors is None or shortest_path_costs is None:
        shortest_path_costs, shortest_path_predecessors = shortest_path(
            graph, return_predecessors=True, directed=directed)
    n = graph.shape[0]
    non_terminals = list(set(range(n)) - set(terminals))
    terminal_positions = {} # maps vertex indices to positions in terminals
    for i, terminal in enumerate(terminals):
        terminal_positions[terminal] = i
    steiner_edges = []
    # For the keys in these dicts, we need to convert the Numpy arrays to
    # strings to make them hashable. Apparently this is the fastest way to get
    # a hashable copy of an array.
    best_candidates = {}    # maps (vertex, remaining) to (cost, vertex)
                            # (where the 2nd vertex is the next to connect to
                            # achieve the specified cost)
    best_splits = {} # maps (vertex, remaining) to (cost, terminals)
    num_terminals = len(terminals)
    start_terminal_index = min(start_terminal_index, num_terminals - 1)
    # set all terminals except start to True (they're all remaining)
    remaining = np.ones(num_terminals, dtype=np.bool)
    remaining[start_terminal_index] = False

    # Run dynamic programming algorithm
    connect_vertex(terminals[start_terminal_index], remaining)
    retrieve_solution_connect(terminals[start_terminal_index], remaining)

    # Now we need to reconstruct the solution *without* pretending that all
    # nodes are connected via an edge with the weight of their shortest path.
    steiner_nodes = set()
    steiner_tree = lil_matrix((n, n), dtype=graph.dtype)
    def get_real_edge_cost(start, end):
        cost = graph[start, end]
        if cost == 0:
            cost = np.inf
        return cost
    for start_terminal_index, end in steiner_edges:
        real_path = reconstruct_predecessor_path(
            shortest_path_predecessors, start_terminal_index, end)
        for vertex in real_path:
            if vertex not in terminal_positions:
                steiner_nodes.add(vertex)
        for real_start, real_end in pairwise(real_path):
            # Check whether we have a forward path in the original graph.
            edge_cost = get_real_edge_cost(real_start, real_end)
            reverse_edge_cost = get_real_edge_cost(real_end, real_start)
            # If so, and that edge is in fact the smaller of the two, just
            # include it in the output path.
            if edge_cost < reverse_edge_cost:
                steiner_tree[real_start, real_end] = edge_cost
            # Otherwise, the predecessors graph must have been generated in
            # an undirected way, and the reverse edge is either the only edge
            # or the smaller of the two. Include the reverse edge instead.
            else:
                steiner_tree[real_end, real_start] = reverse_edge_cost

    steiner_tree = steiner_tree.tocsr()
    return list(steiner_nodes), steiner_tree


def longest_path_in_tree(tree, start_from=0):
    '''
    Finds the longest *undirected* path in a tree using two searches.

    Algorithm:
      1. Run BFS from an arbitrary node (`start_from`). Call the furthest node
         F1.
      2. Starting from F1, run another BFS. Find the furthest node F2.
      3. Return the path between F1 and F2.

    Parameters
    ----------
    tree : sparse matrix
        Input matrix. Must represent a valid tree.
    start_from : int
        Node to start from. This is useful for cases where the graph *contains*
        a tree, but may contain stranded nodes (or other connected components).
        Specifying the start node allows you to determine what connected
        component the algorithm finds the longest path in. (It still assumes
        that that connected component is a valid tree.)

    Returns
    -------
    path : ndarray
        The ordered list of nodes traversed in the longest path, including the
        start/end nodes.
    '''
    furthest_node_1 = breadth_first_order(tree, start_from, directed=False,
                                          return_predecessors=False)[-1]
    search_result, predecessors = breadth_first_order(tree, furthest_node_1,
                                                      directed=False)
    furthest_node_2 = search_result[-1]
    path = reconstruct_predecessor_path(
        predecessors, furthest_node_1, furthest_node_2)
    # Because furthest_node_1 is furthest from 0, a path found from it will
    # often go from a much higher number to 0. While this is technically valid,
    # it's often more aesthetically appealing to have the path go the other
    # direction, so we reverse what we get from tracing predecessors.
    return path[::-1]


def get_incoming_indices(graph, node):
    if isinstance(graph, np.ndarray) or isinstance(graph, ma.masked_array):
        incoming = graph[:, node]
    else:
        incoming = graph.getcol(node)
    return incoming.nonzero()[0]


def get_outgoing_indices(graph, node):
    if isinstance(graph, np.ndarray) or isinstance(graph, ma.masked_array):
        incoming = graph[node, :]
        return incoming.nonzero()[0] # ndarray-like nonzero produces only 1 dim
    else:
        incoming = graph.getrow(node)
        return incoming.nonzero()[1]


class CycleError(Exception):
    def __init__(self):
        super(Exception, self).__init__("Cycle detected; graph is not a DAG")


def topological_sort(tree, algorithm='tarjan'):
    return tarjan_topological_sort(tree)
    

_SORT_MARKS = Enum(['Unvisited', 'Visiting', 'Visited'])
def tarjan_topological_sort(tree):
    sorted_nodes = []
    # assumes a square matrix, which a graph should be
    marks = [_SORT_MARKS.Unvisited] * tree.shape[0]

    def visit(node):
        if isinstance(node, np.ndarray):
            raise Exception()
        if marks[node] == _SORT_MARKS.Visiting:
            raise CycleError()
        if marks[node] != _SORT_MARKS.Visited:
            marks[node] = _SORT_MARKS.Visiting
            for child in get_outgoing_indices(tree, node):
                visit(child)
            marks[node] = _SORT_MARKS.Visited
            sorted_nodes.append(node)

    next_node = 0
    try:
        while True:
            visit(next_node)
            next_node = marks.index(_SORT_MARKS.Unvisited)
    except ValueError: # Unvisited was not found -> all have been visited
        pass

    sorted_nodes.reverse()
    return sorted_nodes


class DummyLabelEncoder(LabelEncoder):
    def fit(self, y):
        return self
    
    def fit_transform(self, y):
        return y
    
    def transform(self, y):
        return y

    def inverse_transform(self, y):
        return y


class AutoWeightedVotingClassifier(VotingClassifier):
    """
    Voting classifier that fits the classifier weights automatically depending
    on how accurate each is on the training data.
    """
    def __init__(self, estimators, voting='hard', weights=None,
                 score_fn=accuracy, score_probas=False, autofit_weights=True):
        self.score_fn = score_fn
        self.score_probas = score_probas
        self.autofit_weights = autofit_weights
        super(AutoWeightedVotingClassifier, self).__init__(estimators,
                                                           voting, weights)

        # If the original estimators were fitted, allow using the classifier
        # even before fitting.
        self.estimators_ = self.named_estimators.values()
        self.le_ = DummyLabelEncoder()

    def fit_weights(self, X, y):
        weights = []
        for estimator in self.estimators_:
            if self.score_probas and self.voting == 'soft':
                predicted = estimator.predict_proba(X)
            else:
                predicted = estimator.predict(X)
            weights.append(self.score_fn(y, predicted))
        self.weights = normalize([weights], 'l1')[0]
        return self.weights

    def fit(self, X, y):
        retval = super(AutoWeightedVotingClassifier, self).fit(X, y)
        if self.autofit_weights:
            self.fit_weights(X, y)
        return retval


def make_logistic_score(L=1.0, k=1.0, x0=0.0):
    """
    Makes a score function with parameters L, k, and x0. The score function,
    given a list of reference values and a corresponding array of test class
    probabilities, returns the sum of the logistic function applied to all those
    probabilities, where the logistic function is parameterized by L, k, and x0.
    """
    def logistic_score(reference, test_probs):
        probs_of_correct_answer = test_probs[np.arange(len(reference)),
                                             reference]
        return np.sum(L / (1.0 + np.exp(-k * (probs_of_correct_answer - x0))))
    return logistic_score


def prob_sum_score(reference, test_probs):
    probs_of_correct_answer = test_probs[np.arange(len(reference)), reference]
    return np.sum(probs_of_correct_answer)
