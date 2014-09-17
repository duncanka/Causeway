import numpy as np

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
