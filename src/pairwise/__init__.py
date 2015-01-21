from gflags import FLAGS

from data import CausationInstance
from pipeline import Stage
from util.metrics import ClassificationMetrics

# Define a bunch of shared functions that are used by various stages in the
# pipeline.

class PairwiseCausalityStage(Stage):
    def __init__(self, print_test_instances, *args, **kwargs):
        self.print_test_instances = print_test_instances
        super(PairwiseCausalityStage, self).__init__(*args, **kwargs)

    def _begin_evaluation(self):
        self.all_instances_metrics = ClassificationMetrics(finalize=False)
        if self.print_test_instances:
            self.tp_pairs, self.fp_pairs, self.fn_pairs = [], [], []
        else:
            self.tp_pairs, self.fp_pairs, self.fn_pairs = None, None, None

    def _complete_evaluation(self):
        self.all_instances_metrics._finalize_counts()
        if self.print_test_instances:
            PairwiseCausalityStage.print_instances_by_eval_result(
                self.tp_pairs, self.fp_pairs, self.fn_pairs)
        del self.tp_pairs
        del self.fp_pairs
        del self.fn_pairs
        all_instances_metrics = self.all_instances_metrics
        del self.all_instances_metrics
        return all_instances_metrics

    @staticmethod
    def starts_before(token_1, token_2):
        # None, if present as an argument, should be second.
        return token_2 is None or (
            token_1 is not None and
            token_2.start_offset > token_1.start_offset)

    @staticmethod
    def normalize_order(token_pair):
        '''
        Normalizes the order of a token pair so that the earlier one in the
        sentence is first in the pair.
        '''
        if PairwiseCausalityStage.starts_before(*token_pair):
            return tuple(token_pair)
        else:
            return (token_pair[1], token_pair[0])

    @staticmethod
    def match_causation_pairs(expected_pairs, found_pairs, tp_pairs, fp_pairs,
                              fn_pairs, metrics):
        '''
        Match expected and predicted cause/effect pairs from a single sentence.
        expected_pairs and found_pairs are lists of Token tuples.
        *_instances are all lists in which to record the pairs of various sorts
        for later examination (ignored for any that are None).
        '''
        tp, fp, fn = 0, 0, 0
        found_pairs = [PairwiseCausalityStage.normalize_order(pair)
                       for pair in found_pairs]
        expected_pairs = [PairwiseCausalityStage.normalize_order(pair)
                          for pair in expected_pairs]

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
    def print_instances_by_eval_result(tp_pairs, fp_pairs, fn_pairs):
        for pairs, pair_type in zip(
            [tp_pairs, fp_pairs, fn_pairs],
            ['True positives', 'False positives', 'False negatives']):
            print pair_type + ':'
            for pair in pairs:
                # If only one argument is present, it'll be in position 0.
                sentence = pair[0].parent_sentence
                print '    %s ("%s" / "%s")' % (
                    sentence.original_text.replace('\n', ' '),
                    pair[0].original_text,
                    pair[1].original_text if pair[1] else None)

            print '\n'
