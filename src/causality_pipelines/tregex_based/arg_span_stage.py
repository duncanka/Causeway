from gflags import FLAGS
from itertools import permutations, chain
import logging

from causality_pipelines import IAAEvaluator
from pipeline import Stage
from pipeline.models import Model

class ArgSpanModel(Model):
    def __init__(self, *args, **kwargs):
        super(ArgSpanModel, self).__init__(*args, **kwargs)
        self.connectives_allowed_in_arg = set()

    def _train_model(self, sentences):
        for sentence in sentences:
            for pc in sentence.possible_causations:
                if pc.true_causation_instance:
                    for token in pc.connective:
                        if (token in pc.true_causation_instance.cause or
                            token in pc.true_causation_instance.effect):
                            for pattern in pc.matching_patterns:
                                self.connectives_allowed_in_arg.add(
                                    (pattern, token.lemma))

    def reset(self):
        self.connectives_allowed_in_arg = set()

    def test(self, sentences):
        all_expanded_args = [[] for _ in sentences]
        for sentence, sentence_expanded_args in zip(sentences,
                                                    all_expanded_args):
            for pc in sentence.possible_causations:
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
                sentence_expanded_args.append(expanded_args)
        return all_expanded_args

    def expand_argument(self, pc, argument_head,
                        other_argument_head):
        argument_tokens = self.expand_argument_from_token(
            pc, argument_head, other_argument_head, set())
        return sorted(argument_tokens, key=lambda token: token.index)

    def expand_argument_from_token(self, pc, argument_token,
                                   other_argument_head, visited):
        # print "    Expanding", argument_token
        # is_first_token = not visited
        # Use a set to represent the argument tokens in case, in the process of
        # following a dependency cycle, we re-encounter the same node twice.
        expanded_tokens = set([argument_token])
        for child_token in pc.sentence.get_children(argument_token, '*'):
            # Don't revisit already-visited nodes, even if we've come back to
            # them through a dependency cycle.
            if child_token in visited:
                continue
            visited.add(child_token)

            if child_token is other_argument_head or child_token in pc.connective:
                continue

            expanded_tokens.update(
                self.expand_argument_from_token(pc, child_token,
                                                other_argument_head, visited))

        return expanded_tokens

    def expand_argument_from_token_old(self, pc, argument_token,
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
        super(ArgSpanStage, self).__init__(name=name, model=ArgSpanModel())

    def _label_instance(self, document, sentence, predicted_args):
        assert len(sentence.possible_causations) == len(predicted_args)
        for pc, args in zip(sentence.possible_causations, predicted_args):
            pc.cause, pc.effect = args

    def _make_evaluator(self):
        return IAAEvaluator(False, False, FLAGS.args_print_test_instances,
                            True, True, 'possible_causations')
