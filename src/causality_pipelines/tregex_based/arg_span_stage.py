from gflags import DEFINE_bool, FLAGS, DuplicateFlagError
from itertools import permutations
import logging

from causality_pipelines import PossibleCausation, IAAEvaluator
from pipeline import Stage
from pipeline.models import Model

try:
    DEFINE_bool('arg_span_print_test_instances', False,
                'Whether to print differing IAA results during evaluation')
except DuplicateFlagError as e:
    logging.warn('Ignoring redefinition of flag %s' % e.flagname)


class ArgSpanModel(Model):
    def __init__(self, *args, **kwargs):
        super(ArgSpanModel, self).__init__(*args, **kwargs)

    def train(self, possible_causations):
        pass

    def test(self, possible_causations):
        for pc in possible_causations:
            expanded_args = []
            for arg, other_arg in permutations([pc.cause, pc.effect]):
                if arg is None:
                    expanded_args.append(None)
                else:
                    assert len(arg) == 1
                    expanded_arg = self.expand_argument(
                        pc.sentence, pc.connective, arg[0], other_arg[0])
                    if pc.sentence.get_head(expanded_arg) is not arg[0]:
                        logging.warn(
                            'Head changed after expanding args: %s became %s in'
                            ' sentence: "%s"' % (arg, expanded_arg,
                                                 pc.sentence.original_text))
                    expanded_args.append(expanded_arg)
            pc.cause, pc.effect = expanded_args

    @staticmethod
    def expand_argument(sentence, connective, argument_head,
                        other_argument_head):
        # print "Expanding argument", argument_head, "in", sentence.original_text, "(connective: %s)" % connective
        argument_tokens = ArgSpanModel.expand_argument_from_token(
            sentence, connective, argument_head, other_argument_head, set())
        return sorted(argument_tokens, key=lambda token: token.index)

    @staticmethod
    def expand_argument_from_token(sentence, connective, argument_token,
                                   other_argument_head, visited):
        # print "    Expanding", argument_token
        # Use a set to represent the argument tokens in case, in the process of
        # following a dependency cycle, we re-encounter the same node twice.
        is_first_token = not visited
        expanded_tokens = set([argument_token])
        for edge_label, child_token in sentence.get_children(argument_token):
            # Don't revisit already-visited nodes, even if we've come back to
            # them through a dependency cycle.
            if child_token in visited:
                continue

            # Don't expand to conjuncts of the original argument head.
            if is_first_token and edge_label in ['conj', 'cc']:
                continue

            # Connective words that are below an argument word are usually part
            # of the grammaticalization of the connective's argument structure,
            # not part of the argument itself. (Adverbial modifiers are an
            # exception.)
            if child_token in connective and edge_label != 'advmod':
                continue

            # Don't add or recurse into the other argument.
            if child_token is other_argument_head:
                continue

            visited.add(child_token)
            expanded_tokens.update(
                ArgSpanModel.expand_argument_from_token(
                    sentence, connective, child_token, other_argument_head,
                    visited))

        return expanded_tokens


class ArgSpanStage(Stage):
    def __init__(self, name):
        super(ArgSpanStage, self).__init__(
            name=name,
            models=ArgSpanModel(part_type=PossibleCausation))

    def _extract_parts(self, sentence, is_train):
        return sentence.possible_causations

    def _make_evaluator(self):
        # TODO: provide both pairwise and non-pairwise stats
        return IAAEvaluator(False, False, FLAGS.arg_span_print_test_instances,
                            True, True, 'possible_causations')
