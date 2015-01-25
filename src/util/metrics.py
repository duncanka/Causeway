import copy
from gflags import DEFINE_bool, FLAGS, DuplicateFlagError
# import itertools
import logging
import numpy as np
from nltk.metrics import confusionmatrix

try:
    DEFINE_bool('metrics_log_raw_counts', False,
                "Log raw counts (TP, FP, etc.) for evaluation or IAA metrics.");
except DuplicateFlagError as e:
    logging.warn('Ignoring redefinition of flag %s' % e.flagname)

safe_divisor = lambda divisor: divisor if divisor != 0 else float('nan')

class ClassificationMetrics(object):
    def __init__(self, tp=0, fp=0, fn=0, tn=None, finalize=True):
        # Often there is no well-defined concept of a true negative, so it
        # defaults to undefined.
        self._tp = tp
        self._fp = fp
        self._tn = tn
        self._fn = fn

        if finalize:
            self._finalize_counts()
        else:
            self._finalized = False

        #assert tp >= 0 and fp >= 0 and fn >= 0 and (tn is None or tn >= 0), (
        #    'Invalid raw metrics values (%s)' % ((tp, fp, fn, tn),))

    def _finalize_counts(self):
        tp = float(self._tp)
        self._precision = tp / safe_divisor(tp + self._fp)
        self._recall = tp / safe_divisor(tp + self._fn)
        self._f1 = 2 * tp / safe_divisor(2 * tp + self._fp + self._fn)
        if self._tn is not None:
            self._accuracy = (tp + self._tn) / safe_divisor(
                tp + self._tn + self._fp + self._fn)
        else:
            self._accuracy = float('nan')
            self._tn = self._accuracy
        self._finalized = True

    def __str__(self):
        if not self._finalized:
            self._finalize_counts()

        if FLAGS.metrics_log_raw_counts:
            return ('TP: %g\n'
                    'TN: %g\n'
                    'FP: %g\n'
                    'FN: %g\n'
                    'Accuracy: %g\n'
                    'Precision: %g\n'
                    'Recall: %g\n'
                    'F1: %g') % (
                        self._tp, self._tn, self._fp, self._fn, self._accuracy,
                        self._precision, self._recall, self._f1)
        else:
            return ('Accuracy: %g\n'
                    'Precision: %g\n'
                    'Recall: %g\n'
                    'F1: %g') % (
                        self._accuracy, self._precision, self._recall,
                        self._f1)

    @staticmethod
    def average(metrics_list, ignore_nans=True):
        '''
        Averaging produces a technically non-sensical ClassificationMetrics
        object: the usual relationships do not hold between the properties.
        To get around this, we manually modify the underlying attributes, then
        reassure the object that it's been finalized.
        '''
        avg = ClassificationMetrics(0, 0, 0, None, False)
        property_names = (ClassificationMetrics.MUTABLE_PROPERTY_NAMES +
                          ClassificationMetrics.DERIVED_PROPERTY_NAMES)
        for property_name in property_names:
            underlying_property_name = '_' + property_name
            values = [getattr(m, underlying_property_name)
                      for m in metrics_list]
            if ignore_nans:
                values = [v for v in values if not np.isnan(v)]
            setattr(avg, underlying_property_name,
                    sum(values) / safe_divisor(float(len(values))))
        avg._finalized = True
        return avg
    
    ''' We need a bunch of extra functions to support property creation. '''

    @staticmethod
    def make_mutable_getter(property_name):
        def getter(self):
            if not self._finalized:
                self._finalize_counts()
            return getattr(self, '_' + property_name)
        return getter

    @staticmethod
    def make_derived_getter(property_name):
        return lambda self: getattr(self, '_' + property_name)

    @staticmethod
    def make_real_setter(property_name):
        def setter(self, value):
            setattr(self, '_' + property_name, value)
            self._finalized = False
        return setter

    @staticmethod
    def make_derived_setter(property_name):
        def setter(self, value):
            raise ValueError('%s property is not directly modifiable'
                             % property_name)
        return setter

    MUTABLE_PROPERTY_NAMES = ['tp', 'fp', 'fn', 'tn']
    DERIVED_PROPERTY_NAMES = ['accuracy', 'precision', 'recall', 'f1']

for property_name in (ClassificationMetrics.MUTABLE_PROPERTY_NAMES
                      + ClassificationMetrics.DERIVED_PROPERTY_NAMES):
    if property_name in ClassificationMetrics.MUTABLE_PROPERTY_NAMES:
        getter = ClassificationMetrics.make_derived_getter(property_name)
        setter = ClassificationMetrics.make_real_setter(property_name)
    else: # derived property
        getter = ClassificationMetrics.make_mutable_getter(property_name)
        setter = ClassificationMetrics.make_derived_setter(property_name)
    setattr(ClassificationMetrics, property_name, property(getter, setter))

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
        try:
            log_metrics = kwargs.pop('metrics')
        except KeyError:
            log_metrics = False
        pp = super(ConfusionMatrix, self).pp(*args, **kwargs)
        if (len(args) > 4 and args[4] == True) or log_metrics:
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
