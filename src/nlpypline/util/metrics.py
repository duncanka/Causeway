from __future__ import absolute_import
import copy
from gflags import DEFINE_bool, FLAGS, DuplicateFlagError
import logging
import numpy as np
from nltk.metrics import confusionmatrix
from nlpypline.util.scipy import add_rows_and_cols_to_matrix
from nlpypline.util import floats_same_or_nearly_equal

try:
    DEFINE_bool('metrics_log_raw_counts', False,
                "Log raw counts (TP, FP, etc.) for evaluation or IAA metrics.");
except DuplicateFlagError as e:
    logging.warn('Ignoring flag redefinitions; assuming module reload')


def safe_divide(dividend, divisor):
    if divisor != 0:
        return float(dividend) / divisor
    elif dividend == 0:
        return 0.0
    else:
        return np.nan

def f1(precision, recall):
    return safe_divide(2 * precision * recall, precision + recall)

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

    def __add__(self, other):
        summed = copy.copy(self)
        summed._tp += other._tp
        summed._fp += other._fp
        summed._fn += other._fn
        if summed._tn is None or other._tn is None:
            summed._tn = None
        else:
            summed._tn += other._tn
        summed._finalized = False
        return summed

    def _finalize_counts(self):
        tp = float(self._tp)
        self._precision = safe_divide(tp, tp + self._fp)
        self._recall = safe_divide(tp, tp + self._fn)
        self._f1 = f1(self._precision, self._recall)
        if self._tn is not None:
            self._accuracy = safe_divide(tp + self._tn,
                                         tp + self._tn + self._fp + self._fn)
        else:
            self._accuracy = np.nan
            self._tn = np.nan
        self._finalized = True

    def __repr__(self):
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

    def __eq__(self, other):
        return (isinstance(other, ClassificationMetrics)
                and floats_same_or_nearly_equal(self._tp, other._tp)
                and floats_same_or_nearly_equal(self._fp, other._fp)
                and floats_same_or_nearly_equal(self._fn, other._fn)
                and floats_same_or_nearly_equal(self._tn, other._tn))

    def __ne__(self, other):
        return not self.__eq__(other)

    @staticmethod
    def average(metrics_list, ignore_nans=True):
        '''
        Averaging produces a technically non-sensical ClassificationMetrics
        object: the usual relationships do not hold between the properties.
        To get around this, we manually modify the underlying attributes, then
        reassure the object that it's been finalized.
        '''
        for metrics in metrics_list:
            if not metrics._finalized:
                metrics._finalize_counts()

        avg = ClassificationMetrics(0, 0, 0, None, False)
        property_names = (ClassificationMetrics.MUTABLE_PROPERTY_NAMES +
                          ClassificationMetrics.DERIVED_PROPERTY_NAMES)
        for property_name in property_names:
            underlying_property_name = '_' + property_name
            values = [getattr(m, underlying_property_name)
                      for m in metrics_list]
            if ignore_nans:
                values = [v for v in values if not np.isnan(v)]
            # TN nans are meaningful, so if that's all we got, we keep them.
            if property_name in ['tn', 'accuracy'] and not values:
                attr_val = np.nan
            else:
                attr_val = safe_divide(sum(values), len(values))
            setattr(avg, underlying_property_name, attr_val)

        avg._finalized = True
        return avg

    ''' We need a bunch of extra functions to support property creation. '''

    @staticmethod
    def _make_mutable_getter(property_name):
        def getter(self):
            if not self._finalized:
                self._finalize_counts()
            return getattr(self, '_' + property_name)
        return getter

    @staticmethod
    def _make_derived_getter(property_name):
        return lambda self: getattr(self, '_' + property_name)

    @staticmethod
    def _make_real_setter(property_name):
        def setter(self, value):
            setattr(self, '_' + property_name, value)
            self._finalized = False
        return setter

    @staticmethod
    def _make_derived_setter(property_name):
        def setter(self, value):
            raise ValueError('%s property is not directly modifiable'
                             % property_name)
        return setter

    MUTABLE_PROPERTY_NAMES = ['tp', 'fp', 'fn', 'tn']
    DERIVED_PROPERTY_NAMES = ['accuracy', 'precision', 'recall', 'f1']

for property_name in ClassificationMetrics.MUTABLE_PROPERTY_NAMES:
    getter = ClassificationMetrics._make_derived_getter(property_name)
    setter = ClassificationMetrics._make_real_setter(property_name)
    setattr(ClassificationMetrics, property_name, property(getter, setter))
for property_name in ClassificationMetrics.DERIVED_PROPERTY_NAMES:
    getter = ClassificationMetrics._make_mutable_getter(property_name)
    setter = ClassificationMetrics._make_derived_setter(property_name)
    setattr(ClassificationMetrics, property_name, property(getter, setter))


def diff_binary_vectors(predicted, gold, count_tns=True):
    # Make sure np.where works properly
    predicted = np.array(predicted)
    gold = np.array(gold)

    tp = np.count_nonzero((predicted == 1) & (gold == 1))
    fp = np.count_nonzero((predicted == 1) & (gold == 0))
    fn = np.count_nonzero((predicted == 0) & (gold == 1))
    if count_tns:
        tn = np.count_nonzero((predicted == 0) & (gold == 0))
    else:
        tn = None
    return ClassificationMetrics(tp, fp, fn, tn)


class ConfusionMatrix(confusionmatrix.ConfusionMatrix):
    def __init__(self, *args, **kwargs):
        kwargs['sort_by_count'] = False
        super(ConfusionMatrix, self).__init__(*args, **kwargs)
        self._confusion = np.array(self._confusion)
        self.class_names = self._values
        
    def __add__(self, other):
        # Deal with the possibility of an empty matrix.
        if self._confusion.shape[0] == 0:
            return copy.deepcopy(other)
        elif other._confusion.shape[0] == 0:
            return copy.deepcopy(self)

        # First, create the merged labels list, and figure out what columns
        # we'll need to insert in the respective matrices.
        # Because we've disabled sort by count, _values is already sorted in
        # alphabetical order. 
        i = 0
        j = 0
        self_cols_to_add = [0 for _ in range(len(self._values) + 1)]
        other_cols_to_add = [0 for _ in range(len(other._values) + 1)]
        merged_values = []
        while i < len(self._values) and j < len(other._values):
            if self._values[i] < other._values[j]:
                # I have an item other doesn't. Record where to insert it.
                merged_values.append(self._values[i])
                other_cols_to_add[j] += 1
                i += 1
            elif self._values[i] > other._values[j]:
                # Other has an item I don't. Record where to insert it.
                merged_values.append(other._values[j])
                self_cols_to_add[i] += 1
                j += 1
            else:
                merged_values.append(self._values[i])
                i += 1
                j += 1
        if i < len(self._values): # still some self values left
            merged_values.extend(self._values[i:])
            other_cols_to_add[-1] = len(self._values) - i
        if j < len(other._values): # still some other values left
            merged_values.extend(other._values[j:])
            self_cols_to_add[-1] = len(other._values) - j

        augmented_self_matrix = add_rows_and_cols_to_matrix(self._confusion,
                                                            self_cols_to_add)
        augmented_other_matrix = add_rows_and_cols_to_matrix(other._confusion,
                                                             other_cols_to_add)

        new_matrix = copy.copy(self)
        new_matrix._values = merged_values
        new_matrix.class_names = merged_values
        new_matrix._indices = {val: i for i, val in enumerate(merged_values)}
        new_matrix._confusion = augmented_self_matrix + augmented_other_matrix
        new_matrix._max_conf = max(self._max_conf, other._max_conf)
        new_matrix._total = self._total + other._total
        new_matrix._correct = self._correct + other._correct

        return new_matrix
    
    def __radd__(self, other):
        return other.__add__(self)

    def pretty_format_metrics(self):
        return ('% Agreement: {:.2}\nKappa: {:.2}\n'
                'Micro F1: {:.2}\nMacro F1: {:.2}'.format(
                    self.pct_agreement(), self.kappa(), self.f1_micro(),
                    self.f1_macro()))

    def pretty_format(self, *args, **kwargs):
        """
        Accepts a 'metrics' keyword argument (or fifth positional argument)
        indicating whether to print the agreement metrics, as well.
        """
        try:
            log_metrics = kwargs.pop('metrics')
        except KeyError:
            log_metrics = False
        if self._values:
            pp = super(ConfusionMatrix, self).pretty_format(*args, **kwargs)
            if (len(args) > 4 and args[4] == True) or log_metrics:
                pp += self.pretty_format_metrics()
        else:
            pp = repr(self) # <ConfusionMatrix: 0/0 correct>
        return pp

    def num_agreements(self):
        return self._correct

    def pct_agreement(self):
        return safe_divide(self._correct, self._total)

    def kappa(self):
        if not self._total:
            return float('nan')

        row_totals = np.sum(self._confusion, axis=1)
        col_totals = np.sum(self._confusion, axis=0)
        agree_by_chance = sum([safe_divide(row_total * col_total, self._total)
                               for row_total, col_total
                               in zip(row_totals, col_totals)])
        kappa = safe_divide(self._correct - agree_by_chance,
                            self._total - agree_by_chance)
        return kappa

    def _get_f1_stats_arrays(self):
        # Which axis we call gold and which we call test is pretty arbitrary.
        # It doesn't matter, because F1 is symmetric.
        try:
            tp = self._confusion.diagonal()
            fp = self._confusion.sum(0) - tp
            fn = self._confusion.sum(1) - tp
            return (tp, fp, fn)
        except ValueError:
            logging.warn("Tried to get F1 stats for empty confusion matrix")
            return (np.full((self._confusion.shape[0],), np.nan),) * 3

    def f1_micro(self):
        _, fp, fn = self._get_f1_stats_arrays()
        p_micro = safe_divide(self._correct, self._correct + fp.sum())
        r_micro = safe_divide(self._correct, self._correct + fn.sum())
        return f1(p_micro, r_micro)

    def f1_macro(self):
        tp, fp, fn = self._get_f1_stats_arrays()
        p_macro_fractions = tp / np.sum([tp, fp], axis=0, dtype=float)
        p_macro = np.average(p_macro_fractions)
        r_macro_fractions = tp / np.sum([tp, fn], axis=0, dtype=float)
        r_macro = np.average(r_macro_fractions)
        return f1(p_macro, r_macro)


class AccuracyMetrics(object):
    def __init__(self, correct, incorrect):
        self.correct = correct
        self.incorrect = incorrect
        self.accuracy = safe_divide(correct, correct + incorrect)

    def pretty_format(self):
        if FLAGS.metrics_log_raw_counts:
            return ('Correct: {:}\nIncorrect: {:}\n% Agreement: {:}'
                    .format(self.correct, self.incorrect, self.accuracy))
        else:
            return '% Agreement: {:}'.format(self.accuracy)

    def __repr__(self):
        return self.pretty_format()

    def __add__(self, other):
        return AccuracyMetrics(self.correct + other.correct,
                               self.incorrect + other.incorrect)

    @staticmethod
    def average(metrics):
        new_metrics = object.__new__(AccuracyMetrics)
        new_metrics.correct = np.mean([m.correct for m in metrics])
        new_metrics.incorrect = np.mean([m.incorrect for m in metrics])
        new_metrics.accuracy = np.mean([m.accuracy for m in metrics])
        return new_metrics

    def __eq__(self, other):
        return (isinstance(other, AccuracyMetrics)
                and self.correct == other.correct
                and self.incorrect == other.incorrect
                # Extra check to make sure averages work right
                and floats_same_or_nearly_equal(self.accuracy, other.accuracy))

    def __ne__(self, other):
        return not self.__eq__(other)

