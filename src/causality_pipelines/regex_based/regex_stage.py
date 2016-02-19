from collections import defaultdict
from gflags import DEFINE_bool, FLAGS, DuplicateFlagError
import logging
import re
import time

from causality_pipelines import PossibleCausation, IAAEvaluator
from data import StanfordParsedSentence
from pipeline import Stage
from pipeline.models import Model
from util import Enum

try:
    DEFINE_bool('regex_print_patterns', False,
                'Whether to print all connective patterns')
    DEFINE_bool('regex_print_test_instances', False,
                'Whether to print differing IAA results during evaluation')
    DEFINE_bool('regex_include_pos', True,
                'Whether to include POS tags in the strings matched by regex')

except DuplicateFlagError as e:
    logging.warn('Ignoring redefinition of flag %s' % e.flagname)


class RegexConnectiveModel(Model):
    def __init__(self, *args, **kwargs):
        super(RegexConnectiveModel, self).__init__(*args, **kwargs)
        self.regexes = []

    def _train_model(self, sentences):
        self.regexes = [
            (re.compile(pattern), matching_groups)
            for pattern, matching_groups in self._extract_patterns(sentences)]

    def test(self, sentences):
        logging.info('Tagging possible connectives...')
        start_time = time.time()

        all_possible_causations = [[] for _ in sentences]
        for sentence, possible_causations in zip(sentences,
                                                 all_possible_causations):
            tokens = sentence.tokens[1:] # skip ROOT
            if FLAGS.regex_include_pos:
                lemmas_to_match = ['%s/%s' % (token.lemma, token.get_gen_pos())
                                 for token in tokens]
            else:
                lemmas_to_match = [token.lemma for token in tokens]
            # Remember bounds of tokens so that we can recover the correct
            # tokens from regex matches.
            token_bounds = []
            # Final space eases matching
            string_to_match = ' '.join(lemmas_to_match) + ' '
            next_start = 0
            for lemma in lemmas_to_match:
                token_bounds.append((next_start, next_start + len(lemma)))
                next_start += len(lemma) + 1

            # More than one pattern may match a given connective. We record
            # which patterns matched which sets of connective words.
            matches = defaultdict(list)
            for regex, matching_group_indices in self.regexes:
                match = regex.search(string_to_match)
                while match is not None:
                    # We need to add 1 to indices to account for root.
                    token_indices = tuple(token_bounds.index(match.span(i)) + 1
                                          for i in matching_group_indices)
                    matches[token_indices].append(regex.pattern)
                    # Skip past the first token that matched to start looking
                    # for the next match. This ensures that we won't match the
                    # same connective twice with this pattern.
                    # (We start from the end of the first group *after* the
                    # pattern start group.)
                    match = regex.search(string_to_match, pos=match.span(2)[1])

            for token_indices, matching_patterns in matches.items():
                connective_tokens = [sentence.tokens[i] for i in token_indices]
                true_causation_instance = None
                for causation_instance in sentence.causation_instances:
                    if causation_instance.connective == connective_tokens:
                        true_causation_instance = causation_instance

                possible_causation = PossibleCausation(
                    sentence, matching_patterns, connective_tokens,
                    true_causation_instance)
                possible_causations.append(possible_causation)

        elapsed_seconds = time.time() - start_time
        logging.info("Done tagging possible connectives in %0.2f seconds"
                     % elapsed_seconds)
        return all_possible_causations

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

    CONNECTIVE_INTERJECTION_PATTERN = ARG_WORDS_PATTERN = '([\S]+ )+?'
    # Pattern can start after another word, or @ start of sentence
    PATTERN_START = '(^| )'
    TokenTypes = Enum(['Connective', 'Cause', 'Effect']) # Also possible: None
    @staticmethod
    def _get_pattern(sentence, connective_tokens, cause_tokens,
                     effect_tokens):
        connective_capturing_groups = []
        pattern = RegexConnectiveModel.PATTERN_START
        next_group_index = 2 # whole match is 0, and pattern start will add 1

        previous_token_type = None
        connective_tokens.sort(key=lambda token: token.index) # just in case
        next_connective_index = 0
        for token in sentence.tokens[1:]:
            if (next_connective_index < len(connective_tokens) and
                token.index == connective_tokens[next_connective_index].index):
                # We ensure above that every token lemma in the tested string
                # has a space after it, even the last token, so space is safe.
                if FLAGS.regex_include_pos:
                    pattern += '(%s/%s) ' % (token.lemma, token.get_gen_pos())
                else:
                    pattern += '(%s) ' % token.lemma
                previous_token_type = (
                    RegexConnectiveModel.TokenTypes.Connective)
                connective_capturing_groups.append(next_group_index)
                next_group_index += 1
                next_connective_index += 1
            else:
                if token in cause_tokens:
                    token_type = RegexConnectiveModel.TokenTypes.Cause
                elif token in effect_tokens:
                    token_type = RegexConnectiveModel.TokenTypes.Effect
                else:
                    token_type = None

                if previous_token_type != token_type:
                    if token_type is None:
                        # It's possible for a connective to be interrupted by a
                        # word that's not consistent enough to make it count as
                        # a connective token (e.g., a determiner).
                        if (token.index > connective_tokens[0].index
                            and next_connective_index < len(connective_tokens)):
                            # We're in the middle of the connective
                            pattern += (RegexConnectiveModel.
                                        CONNECTIVE_INTERJECTION_PATTERN)
                            next_group_index += 1
                    else: # we've transitioned from non-argument to argument
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


class RegexConnectiveStage(Stage):
    def __init__(self, name):
        super(RegexConnectiveStage, self).__init__(
            name=name, model=RegexConnectiveModel())

    def _make_evaluator(self):
        return IAAEvaluator(False, False, FLAGS.regex_print_test_instances,
                            False, True, 'possible_causations')

    produced_attributes = ['possible_causations']

    def _label_instance(self, document, sentence, possible_causations):
        sentence.possible_causations = possible_causations
