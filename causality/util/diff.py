# Based on http://en.wikibooks.org/wiki/Algorithm_Implementation/Strings/Longest_common_subsequence#Python

class SequenceDiff(object):
    def __init__(self, a, b):
        self.a = a
        self.b = b
        self.lcs = self.compute_lcs_matrix(a, b)
        self.pairs = self.compute_pairs(self.lcs, a, b)

    @staticmethod
    def compute_lcs_matrix(a, b):
        m = len(a)
        n = len(b)
        # An (m+1) times (n+1) matrix
        lcs_matrix = [[0 for j in range(n+1)] for i in range(m+1)]
        for i in range(1, m+1):
            for j in range(1, n+1):
                if a[i-1] == b[j-1]:
                    lcs_matrix[i][j] = lcs_matrix[i-1][j-1] + 1
                else:
                    lcs_matrix[i][j] = max(lcs_matrix[i][j-1],
                                           lcs_matrix[i-1][j])
        return lcs_matrix

    @staticmethod
    def compute_pairs(lcs_matrix, a, b):
        return SequenceDiff._compute_pairs_helper(
            lcs_matrix, a, b, len(a), len(b))

    @staticmethod
    def _compute_pairs_helper(lcs_matrix, a, b, i, j):
        if i > 0 and j > 0 and a[i-1] == b[j-1]:
            sub_pairs = SequenceDiff._compute_pairs_helper(
                lcs_matrix, a, b, i-1, j-1)
            return sub_pairs + [(a[i-1], b[j-1])]
        else:
            if j > 0 and (i == 0 or lcs_matrix[i][j-1] >= lcs_matrix[i-1][j]):
                sub_pairs = SequenceDiff._compute_pairs_helper(
                    lcs_matrix, a, b, i, j-1)
                return sub_pairs + [(None, b[j-1])]
            elif i > 0 and (j == 0 or lcs_matrix[i][j-1] < lcs_matrix[i-1][j]):
                sub_pairs = SequenceDiff._compute_pairs_helper(
                    lcs_matrix, a, b, i-1, j)
                return sub_pairs + [(a[i-1], None)]
            else:
                return []
