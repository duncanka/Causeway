import copy
import itertools
import numpy as np
from nltk.metrics import confusionmatrix

printer_indent_level = 0

class ClassificationMetrics(object):
    def __init__(self, tp, fp, fn, tn=None):
        # Often there is no well-defined concept of a true negative, so it
        # defaults to undefined.
        self.tp = tp
        self.fp = fp
        self.tn = tn
        self.fn = fn

        safe_divisor = lambda divisor: divisor if divisor != 0 else float('nan')

        tp = float(tp)
        self.precision = tp / safe_divisor(tp + fp)
        self.recall = tp / safe_divisor(tp + fn)
        self.f1 = 2 * tp / safe_divisor(2 * tp + fp + fn)
        if tn is not None:
            self.accuracy = (tp + tn) / safe_divisor(tp + tn + fp + fn)
        else:
            self.accuracy = float('nan')
            self.tn = self.accuracy

    def __str__(self):
        indent = '   ' * printer_indent_level
        return ('%sTP: %g\n'
                '%sTN: %g\n'
                '%sFP: %g\n'
                '%sFN: %g\n'
                '%sRecall: %g\n'
                '%sPrecision: %g\n'
                '%sAccuracy: %g\n'
                '%sF1: %g\n') % (
                    indent, self.tp, indent, self.tn, indent, self.fp, indent,
                    self.fn, indent, self.recall, indent, self.precision,
                    indent, self.accuracy, indent, self.f1)

def diff_binary_vectors(predicted, gold):
    # Make sure np.where works properly
    predicted = np.array(predicted)
    gold = np.array(gold)

    tp = np.count_nonzero((predicted == 1) & (gold == 1))
    tn = np.count_nonzero((predicted == 0) & (gold == 0))
    fp = np.count_nonzero((predicted == 1) & (gold == 0))
    fn = np.count_nonzero((predicted == 0) & (gold == 1))
    return ClassificationMetrics(tp, fp, fn, tn)


class ConfusionMatrix(confusionmatrix.ConfusionMatrix):
    def __init__(self, *args, **kwargs):
        kwargs['sort_by_count'] = False
        super(ConfusionMatrix, self).__init__(*args, **kwargs)
        self._confusion = np.array(self._confusion)
        self.class_names = self._values

    def __add__(self, other):
        if set(self._values) != set(other._values):
            raise ValueError(
                "Can't add confusion matrices with different label sets")

        '''
        # NOTE: Apparently I can get around this just by disabling sorting by
        # count above...
        #
        # The NLTK confusion matrix labels can be sorted in order of frequency.
        # The orders may be different, so we normalize the order of the other
        # one's matrix to match self's.
        if self._values == other._values:
            other_reordered_confusion = other._confusion
        else:
            other_reordered_confusion = np.empty(other._confusion.shape)
            remapping = [self._indices[value] for value in other._values]
            for index1, index2 in itertools.product(range(len(remapping)),
                                                    range(len(remapping))):
                other_reordered_confusion[remapping[index1],
                                          remapping[index2]] = (
                    other._confusion[index1, index2])
        '''

        new_matrix = copy.copy(self)
        new_matrix._confusion = self._confusion + other._confusion
        new_matrix._max_conf = max(self._max_conf, other._max_conf)
        new_matrix._total = self._total + other._total
        new_matrix._correct = self._correct + other._correct

        return new_matrix

    def num_agreements(self):
        return self._correct

    def pct_agreement(self):
        return self._correct / float(self._total)

    def kappa(self):
        row_totals = np.sum(self._confusion, axis=1)
        col_totals = np.sum(self._confusion, axis=0)
        total_float = float(self._total)
        agree_by_chance = sum([(row_total * col_total) / total_float
                               for row_total, col_total
                               in zip(row_totals, col_totals)])
        kappa = (self._correct - agree_by_chance) / (
            self._total - agree_by_chance)
        return kappa
