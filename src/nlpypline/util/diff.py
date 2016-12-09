# Based on http://en.wikibooks.org/wiki/Algorithm_Implementation/Strings/Longest_common_subsequence#Python

class SequenceDiff(object):
    def __init__(self, a, b, comparator=lambda x, y: x == y, sort_by_key=None):
        if sort_by_key:
            a = sorted(a, key=sort_by_key)
            b = sorted(b, key=sort_by_key)
        self.a = a
        self.b = b
        self.comparator = comparator
        self.lcs = self._compute_lcs_matrix(a, b)
        self.pairs = self._compute_pairs(self.lcs, a, b, len(a), len(b))

    def get_matching_pairs(self):
        return [pair for pair in self.pairs
                if pair[0] is not None and pair[1] is not None]

    def get_a_only_elements(self):
        return [pair[0] for pair in self.pairs
                if pair[0] is not None and pair[1] is None]

    def get_b_only_elements(self):
        return [pair[1] for pair in self.pairs
                if pair[0] is None and pair[1] is not None]

    def _compute_lcs_matrix(self, a, b):
        m = len(a)
        n = len(b)
        # An (m+1) times (n+1) matrix
        lcs_matrix = [[0 for j in range(n+1)] for i in range(m+1)]
        for i in range(1, m+1):
            for j in range(1, n+1):
                if self.comparator(a[i-1], b[j-1]):
                    lcs_matrix[i][j] = lcs_matrix[i-1][j-1] + 1
                else:
                    lcs_matrix[i][j] = max(lcs_matrix[i][j-1],
                                           lcs_matrix[i-1][j])
        return lcs_matrix

    def _compute_pairs(self, lcs_matrix, a, b, i, j):
        if i > 0 and j > 0 and self.comparator(a[i-1], b[j-1]):
            sub_pairs = self._compute_pairs(lcs_matrix, a, b, i-1, j-1)
            return sub_pairs + [(a[i-1], b[j-1])]
        else:
            if j > 0 and (i == 0 or lcs_matrix[i][j-1] >= lcs_matrix[i-1][j]):
                sub_pairs = self._compute_pairs(lcs_matrix, a, b, i, j-1)
                return sub_pairs + [(None, b[j-1])]
            elif i > 0 and (j == 0 or lcs_matrix[i][j-1] < lcs_matrix[i-1][j]):
                sub_pairs = self._compute_pairs(lcs_matrix, a, b, i-1, j)
                return sub_pairs + [(a[i-1], None)]
            else:
                return []
