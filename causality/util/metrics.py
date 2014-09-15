printer_indent_level = 0

class ClassificationMetrics(object):
    def __init__(self, tp, fp, fn, tn=0):
        self.tp = tp
        self.fp = fp
        self.tn = max(tn, 0)
        self.fn = fn

        tp = float(tp)
        self.precision = tp / (tp + fp)
        self.recall = tp / (tp + fn)
        self.f1 = 2 * tp / (2 * tp + fp + fn)
        if tn != -1:
            self.accuracy = (tp + tn) / (tp + tn + fp + fn)
        else:
            self.accuracy = float('nan')

    def __str__(self):
        indent = '\t' * printer_indent_level
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
