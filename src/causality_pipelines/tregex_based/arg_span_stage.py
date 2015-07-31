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
        self.connectives_allowed_in_arg = set()

    def train(self, possible_causations):
        for pc in possible_causations:
            if pc.true_causation_instance:
                for token in pc.connective:
                    if (token in pc.true_causation_instance.cause or
                        token in pc.true_causation_instance.effect):
                        for pattern in pc.matching_patterns:
                            self.connectives_allowed_in_arg.add((pattern,
                                                                 token.lemma))

    def test(self, possible_causations):
        for pc in possible_causations:
            expanded_args = []
            for arg, other_arg in permutations([pc.cause, pc.effect]):
                if arg is None:
                    expanded_args.append(None)
                else:
                    # Args only contain one token at this point.
                    expanded_arg = self.expand_argument(
                        pc, arg[0], other_arg[0])
                    if pc.sentence.get_head(expanded_arg) is not arg[0]:
                        logging.warn(
                            'Head changed after expanding args: "%s" became'
                            ' "%s" in sentence: "%s"' %
                            (' '.join([t.original_text for t in arg]),
                             ' '.join([t.original_text for t in expanded_arg]),
                             pc.sentence.original_text))
                    expanded_args.append(expanded_arg)
            pc.cause, pc.effect = expanded_args

    def expand_argument(self, pc, argument_head,
                        other_argument_head):
        argument_tokens = self.expand_argument_from_token(
            pc, argument_head, other_argument_head, set())
        return sorted(argument_tokens, key=lambda token: token.index)

    def expand_argument_from_token(self, pc, argument_token,
                                   other_argument_head, visited):
        # print "    Expanding", argument_token
        is_first_token = not visited
        # Use a set to represent the argument tokens in case, in the process of
        # following a dependency cycle, we re-encounter the same node twice.
        expanded_tokens = set([argument_token])
        for edge_label, child_token in pc.sentence.get_children(argument_token):
            # Don't revisit already-visited nodes, even if we've come back to
            # them through a dependency cycle.
            if child_token in visited:
                continue

            # Don't expand to conjuncts or parataxes of the original argument
            # head.
            if is_first_token and edge_label in ['conj', 'cc', 'parataxis']:
                continue

            # Connective words that are below an argument word are usually part
            # of the grammaticalization of the connective's argument structure,
            # not part of the argument itself. Only allow words that have been
            # seen in training to be allowed by this pattern to be included in
            # the argument.
            if child_token in pc.connective:
                pattern_allows_connective_lemma = False
                for pattern in pc.matching_patterns:
                    if (pattern, child_token.lemma) in (
                        self.connectives_allowed_in_arg):
                        pattern_allows_connective_lemma = True
                        break
                if not pattern_allows_connective_lemma:
                    continue

            # Don't add or recurse into the other argument.
            if child_token is other_argument_head:
                continue

            visited.add(child_token)
            expanded_tokens.update(
                self.expand_argument_from_token(pc, child_token,
                                                other_argument_head, visited))

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
