import copy
from gflags import DEFINE_bool, FLAGS, DuplicateFlagError
import itertools
import numpy as np
from nltk.metrics import confusionmatrix

try:
    DEFINE_bool('metrics_log_raw_counts', False,
                "Log raw counts (TP, FP, etc.) for evaluation or IAA metrics.");
except DuplicateFlagError as e:
    logging.warn('Ignoring redefinition of flag %s' % e.flagname)

safe_divisor = lambda divisor: divisor if divisor != 0 else float('nan')

class ClassificationMetrics(object):
    def __init__(self, tp, fp, fn, tn=None):
        # Often there is no well-defined concept of a true negative, so it
        # defaults to undefined.
        self.tp = tp
        self.fp = fp
        self.tn = tn
        self.fn = fn

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
        if FLAGS.metrics_log_raw_counts:
            return ('TP: %g\n'
                    'TN: %g\n'
                    'FP: %g\n'
                    'FN: %g\n'
                    'Accuracy: %g\n'
                    'Precision: %g\n'
                    'Recall: %g\n'
                    'F1: %g') % (
                        self.tp, self.tn, self.fp, self.fn, self.accuracy,
                        self.precision, self.recall, self.f1)
        else:
            return ('Accuracy: %g\n'
                    'Precision: %g\n'
                    'Recall: %g\n'
                    'F1: %g') % (
                        self.accuracy, self.precision, self.recall, self.f1)


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

    def pp_metrics(self):
        return '% Agreement: {:.2}\nKappa: {:.2}'.format(
            self.pct_agreement(), self.kappa())

    def pp(self, *args, **kwargs):
        """
        Accepts a 'metrics' keyword argument (or fifth positional argument)
        indicating whether to print the agreement metrics, as well.
        """
        pp = super(ConfusionMatrix, self).pp(*args, **kwargs)
        if (len(args) > 4 and args[4] == True) or kwargs.get('metrics', False):
            pp += self.pp_metrics()
        return pp

    def num_agreements(self):
        return self._correct

    def pct_agreement(self):
        return self._correct / safe_divisor(float(self._total))

    def kappa(self):
        if not self._total:
            return float('nan')

        row_totals = np.sum(self._confusion, axis=1)
        col_totals = np.sum(self._confusion, axis=0)
        total_float = safe_divisor(float(self._total))
        agree_by_chance = sum([(row_total * col_total) / total_float
                               for row_total, col_total
                               in zip(row_totals, col_totals)])
        kappa = (self._correct - agree_by_chance) / safe_divisor(
            self._total - agree_by_chance)
        return kappa
