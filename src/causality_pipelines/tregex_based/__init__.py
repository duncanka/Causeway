from gflags import FLAGS

from data import CausationInstance
from pipeline import Stage, Evaluator
from util.metrics import ClassificationMetrics

# Define a bunch of shared functions that are used by various stages in the
# pipeline.

class PossibleCausation(object):
    def __init__(self, arg1, arg2, matching_pattern, label):
        self.arg1 = arg1
        self.arg2 = arg2
        self.matching_pattern = matching_pattern
        self.correct = label

def starts_before(token_1, token_2):
    # None, if present as an argument, should be second.
    return token_2 is None or (
        token_1 is not None and
        token_2.start_offset > token_1.start_offset)

def normalize_order(token_pair):
    '''
    Normalizes the order of a token pair so that the earlier one in the
    sentence is first in the pair.
    '''
    if starts_before(*token_pair):
        return tuple(token_pair)
    else:
        return (token_pair[1], token_pair[0])


class PairwiseCausalityEvaluator(Evaluator):
    def __init__(self, print_test_instances):
        self._all_instances_metrics = ClassificationMetrics(finalize=False)
        self.print_test_instances = print_test_instances
        if print_test_instances:
            self._tp_pairs, self._fp_pairs, self._fn_pairs = [], [], []
        else:
            self._tp_pairs, self._fp_pairs, self._fn_pairs = None, None, None

    def complete_evaluation(self):
        self._all_instances_metrics._finalize_counts()
        if self.print_test_instances:
            self._print_instances_by_eval_result(
                self._tp_pairs, self._fp_pairs, self._fn_pairs)
        self._tp_pairs, self._fp_pairs, self._fn_pairs = None, None, None
        all_instances_metrics = self._all_instances_metrics
        self._all_instances_metrics = None
        return all_instances_metrics

    @staticmethod
    def _match_causation_pairs(expected_pairs, found_pairs, tp_pairs, fp_pairs,
                              fn_pairs, metrics):
        '''
        Match expected and predicted cause/effect pairs from a single sentence.
        expected_pairs and found_pairs are lists of Token tuples. For Tokens to
        match, they must have come from the same original sentence object
        (though this will still work if they came from shallow-copied objects).
        tp_pairs, fp_pairs, and fn_pairs are all lists in which to record the
        pairs of various sorts for later examination (ignored for any that are
        None).
        '''
        tp, fp, fn = 0, 0, 0
        found_pairs = [normalize_order(pair) for pair in found_pairs]
        expected_pairs = [normalize_order(pair) for pair in expected_pairs]

        for found_pair in found_pairs:
            try:
                expected_pairs.remove(found_pair)
                tp += 1
                if tp_pairs is not None:
                    tp_pairs.append(found_pair)
            except ValueError: # found_pair wasn't in expected_pairs
                fp += 1
                if fp_pairs is not None:
                    fp_pairs.append(found_pair)

        if fn_pairs is not None:
            fn_pairs.extend(expected_pairs)
        fn += len(expected_pairs)

        metrics.tp += tp
        metrics.fp += fp
        metrics.fn += fn

        return tp, fp, fn

    @staticmethod
    def _print_instances_by_eval_result(tp_pairs, fp_pairs, fn_pairs):
        for pairs, pair_type in zip(
            [tp_pairs, fp_pairs, fn_pairs],
            ['True positives', 'False positives', 'False negatives']):
            print pair_type + ':'
            for pair in pairs:
                try:
                    sentence = pair[0].parent_sentence
                except AttributeError:
                    sentence = pair[1].parent_sentence
                args = [(('"%s"' % arg.original_text) if arg else None)
                        for arg in pair]
                print ('    %s ("%s" / %s)' % (
                   sentence.original_text.replace('\n', ' '),
                   args[0], args[1])).encode('utf-8')

            print '\n'
