from gflags import DEFINE_bool, FLAGS, DuplicateFlagError
import logging
import re
import time

from causality_pipelines.regex_based import PossibleCausation
from data import ParsedSentence
from pipeline import ClassifierStage
from pipeline.models import Model
from util import Enum

try:
    DEFINE_bool('regex_print_patterns', False,
                'Whether to print all connective patterns')
    DEFINE_bool('regex_print_test_instances', False,
                'Whether to print true positive, false positive, and false'
                ' negative instances after testing')

except DuplicateFlagError as e:
    logging.warn('Ignoring redefinition of flag %s' % e.flagname)


class RegexConnectiveModel(Model):
    def __init__(self, *args, **kwargs):
        super(RegexConnectiveModel, self).__init__(part_type=ParsedSentence,
                                                   *args, **kwargs)
        self.regexes = []

    def train(self, sentences):
        self.regexes = [
            (re.compile(pattern), matching_groups)
            for pattern, matching_groups in self._extract_patterns(sentences)]

    def test(self, sentences):
        logging.info('Tagging possible connectives...')
        start_time = time.time()

        for sentence in sentences:
            sentence.possible_causations = []

            lemmas = [token.lemma for token in sentence.tokens[1:]] # skip ROOT
            # Remember bounds of tokens so that we can recover the correct
            # token identities from regex matches.
            token_bounds = []
            lemmas_string = ' '.join(lemmas) + ' ' # Final space eases matching
            next_start = 0
            for lemma in lemmas:
                token_bounds.append((next_start, next_start + len(lemma)))
                next_start += len(lemma) + 1

            for regex, matching_group_indices in self.regexes:
                match = regex.match(lemmas_string)
                if match:
                    # We need to add 1 to indices to account for root.
                    token_indices = [token_bounds.index(match.span(i)) + 1
                                     for i in matching_group_indices]
                    connective_tokens = [sentence.tokens[i]
                                         for i in token_indices]

                    true_causation_instance = None
                    for causation_instance in sentence.causation_instances:
                        if causation_instance.connective == connective_tokens:
                            true_causation_instance = causation_instance
                    possible_causation = PossibleCausation(
                        regex.pattern, connective_tokens, true_causation_instance)
                    sentence.possible_causations.append(possible_causation)

        elapsed_seconds = time.time() - start_time
        logging.info("Done tagging possible connectives in %0.2f seconds"
                     % elapsed_seconds)

    #####################################
    # Sentence preprocessing
    #####################################

    @staticmethod
    def _filter_sentences_for_pattern(sentences, pattern, connective_lemmas):
        possible_sentence_indices = []
        for i, sentence in enumerate(sentences):
            token_lemmas = [token.lemma for token in sentence.tokens]
            # TODO: Should we filter here by whether there are enough tokens in
            # the sentence to match the rest of the pattern, too?
            if all([connective_lemma in token_lemmas
                    for connective_lemma in connective_lemmas]):
                possible_sentence_indices.append(i)

        return possible_sentence_indices

    #####################################
    # Pattern generation
    #####################################

    ARG_WORDS_PATTERN = '([\S]+ )+'
    START_OF_PATTERN = '(^|([\S]+ ))'
    TokenTypes = Enum(['Connective', 'Cause', 'Effect']) # Also possible: None
    @staticmethod
    def _get_pattern(sentence, connective_tokens, cause_tokens,
                     effect_tokens):
        assert connective_tokens

        connective_capturing_groups = []
        # Pattern can start after another word or @ start of sentence
        pattern = ''
        next_group_index = 1 # whole match is 0

        previous_token_type = None
        # TODO: Is there a more efficient way to do this?
        for token in sentence.tokens:
            if token in connective_tokens:
                # We ensure above that every token lemma in the tested string
                # has a space after it, even the last token, so space is safe.
                pattern += '(%s) ' % token.lemma
                previous_token_type = (
                    RegexConnectiveModel.TokenTypes.Connective)
                connective_capturing_groups.append(next_group_index)
                next_group_index += 1
            else:
                if token in cause_tokens:
                    token_type = RegexConnectiveModel.TokenTypes.Cause
                elif token in effect_tokens:
                    token_type = RegexConnectiveModel.TokenTypes.Effect
                else:
                    token_type = None

                if previous_token_type != token_type and token_type is not None:
                    pattern += RegexConnectiveModel.ARG_WORDS_PATTERN
                    next_group_index += 1
                previous_token_type = token_type

        return pattern, connective_capturing_groups

    @staticmethod
    def _extract_patterns(sentences):
        # TODO: Extend this to work with cases of missing arguments.
        regex_patterns = []
        patterns_seen = set()

        if FLAGS.tregex_print_patterns:
            print 'Patterns:'
        for sentence in sentences:
            for instance in sentence.causation_instances:
                connective = instance.connective
                cause_tokens, effect_tokens = [
                    arg if arg is not None else []
                    for arg in [instance.cause, instance.effect]]

                pattern, connective_capturing_groups = (
                    RegexConnectiveModel._get_pattern(
                        sentence, connective, cause_tokens, effect_tokens))

                if pattern not in patterns_seen:
                    if FLAGS.regex_print_patterns:
                        print pattern.encode('utf-8')
                        print 'Sentence:', sentence.original_text.encode(
                            'utf-8')
                        print
                    patterns_seen.add(pattern)
                    regex_patterns.append((pattern,
                                           connective_capturing_groups))
        return regex_patterns


class RegexConnectiveStage(ClassifierStage):
    def __init__(self, name):
        super(RegexConnectiveStage, self).__init__(
            name=name,
            models=[RegexConnectiveModel()])
        self.print_test_instances = FLAGS.regex_print_test_instances

    PRODUCED_ATTRIBUTES = ['possible_causations']

    def _extract_parts(self, sentence, is_train):
        return [sentence]

    def _begin_evaluation(self):
        super(RegexConnectiveStage, self)._begin_evaluation()
        self._tp_connectives, self._fp_connectives, self._fn_connectives = (
            [], [], [])

    def _evaluate(self, sentences, original_sentences):
        for sentence in sentences:
            expected_connectives = set(
                tuple(sorted(t.index for t in instance.connective))
                for instance in sentence.causation_instances)
            predicted_connectives = set([
                tuple(sorted(t.index for t in pc.connective))
                for pc in sentence.possible_causations])

            for predicted in predicted_connectives:
                if predicted in expected_connectives:
                    if self.print_test_instances:
                        tp_tokens = [sentence.tokens[i] for i in predicted]
                        self._tp_connectives.append(tp_tokens)
                    self.tp += 1
                    expected_connectives.remove(predicted)
                else:
                    if self.print_test_instances:
                        fp_tokens = [sentence.tokens[i] for i in predicted]
                        self._fp_connectives.append(fp_tokens)
                    self.fp += 1
            # Any expected connectives remaining are ones we didn't predict,
            # i.e., false negatives.
            self.fn += len(expected_connectives)
            if self.print_test_instances:
                tn_token_lists = [[sentence.tokens[i] for i in expected]
                               for expected in expected_connectives]
                self._fn_connectives.extend(tn_token_lists)

    def _complete_evaluation(self):
        result = super(RegexConnectiveStage, self)._complete_evaluation()
        if self.print_test_instances:
            self.print_instances_by_eval_result(
                self._tp_connectives, self._fp_connectives, self._fn_connectives)
        del self._tp_connectives
        del self._fp_connectives
        del self._fn_connectives
        return result

    @staticmethod
    def print_instances_by_eval_result(tp_connectives, fp_connectives,
                                       fn_connectives):
        for connectives, pair_type in zip(
            [tp_connectives, fp_connectives, fn_connectives],
            ['True positives', 'False positives', 'False negatives']):
            print pair_type + ':'
            for connective in connectives:
                sentence = connective[0].parent_sentence
                connective_text = ', '.join([t.original_text
                                             for t in connective])
                print ('    %s (%s)' % (
                   sentence.original_text.replace('\n', ' '),
                   connective_text)).encode('utf-8')

            print '\n'
