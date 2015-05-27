from collections import defaultdict
from gflags import DEFINE_integer, FLAGS, DuplicateFlagError
from itertools import product

from causality_pipelines import IAAEvaluator, PossibleCausation
from data import ParsedSentence
import logging
from pipeline import Stage
from pipeline.models import Model

try:
    DEFINE_integer('baseline_parse_radius', 2,
                   'Maximum number of parse links within which to allow the'
                   ' baseline to look for causes/effects')
except DuplicateFlagError as e:
    logging.warn('Ignoring redefinition of flag %s' % e.flagname)


class BaselineModel(Model):
    _STOP_LEMMAS = ['be', 'the', 'a']
    _STOP_POSES = ['MD', 'CC', 'UH', ':', "''", ',' '.']
    
    def __init__(self, save_results_in):
        super(BaselineModel, self).__init__(ParsedSentence)
        self._connectives = set()
        # (connective word tuple, parse path to cause, parse path to effect) ->
        #   list of [# causal, # non-causal]
        self._connective_relation_counts = defaultdict(lambda: [0, 0])
        self._save_results_in = save_results_in

    @staticmethod
    def _get_closest_connective(sentence, cause, effect, connective_lemmas,
                                tokens_for_lemmas):
        connective_tokens = []
        # We have possible tokens in the sentence for each connective lemma.
        # Iterate through them to see if any are close enough to the args.
        for possible_connective_tokens in tokens_for_lemmas:
            # Find closest connective token to either the cause head or the
            # effect head.
            closest_to_cause, cause_distance = sentence.get_closest_of_tokens(
                cause, possible_connective_tokens)
            # TODO: Add dist(closest_to_cause, cause) to cause_distance, so that
            # comparison finds the one with the minimum global distance?
            closest_to_effect, effect_distance = sentence.get_closest_of_tokens(
                effect, possible_connective_tokens)
            if effect_distance < cause_distance:
                closest = closest_to_effect
                distance = cause_distance
            else:
                closest = closest_to_cause
                distance = effect_distance

            # Abort if this connective token is too far away from both args.
            if distance > FLAGS.baseline_parse_radius:
                return None

            connective_tokens.append(closest)

        connective_tokens.sort(key=lambda token: token.index)
        # Make sure the connective lemmas are in the order they're supposed to
        # be in according to the known pattern.
        if tuple(t.lemma for t in connective_tokens) == connective_lemmas:
            return connective_tokens
        else:
            return None

    @staticmethod
    def _get_dependency_path(sentence, token, connective_tokens):
        closest_connective_token, distance = sentence.get_closest_of_tokens(
            token, connective_tokens)
        if distance > FLAGS.baseline_parse_radius: # possibly infinite
            return None

        deps = sentence.extract_dependency_path(closest_connective_token, token,
                                                False)
        return str(deps)
    
    @staticmethod
    def _get_causation_tuple(connective_tokens, cause, effect):
        return (tuple(t.index for t in connective_tokens),
                cause.index, effect.index)

    def _operate_on_sentences(self, sentences, callback):
        '''
        Extract common iteration logic for both training and test: examine all
        connectives for all pairs of content words in all sentences, and for
        each extract the paths. If the connective is present and there are
        paths, call callback.
        '''
        for i, sentence in enumerate(sentences):
            tokens = [t for t in sentence.tokens[1:]
                      if t.lemma not in self._STOP_LEMMAS
                      and t.pos not in self._STOP_POSES]
            for connective_lemmas in self._connectives:
                all_token_pairs = product(tokens, repeat=2)
                tokens_for_lemmas = [
                    [t for t in sentence.tokens if t.lemma == lemma]
                    for lemma in connective_lemmas]
                # Optimization: skip this connective if any one of its words
                # doesn't appear in the sentence.
                if [] in tokens_for_lemmas:
                    continue

                for possible_cause, possible_effect in all_token_pairs:
                    if possible_cause is possible_effect: # ignore self-pairings
                        continue

                    possible_connective = self._get_closest_connective(
                        sentence, possible_cause, possible_effect,
                        connective_lemmas, tokens_for_lemmas)
                    if possible_connective is None:
                        continue

                    path_1, path_2 = [
                        self._get_dependency_path(sentence, token,
                                                  possible_connective)
                        for token in [possible_cause, possible_effect]]
                    if path_1 and path_2:
                        callback(i, connective_lemmas, possible_connective,
                                 possible_cause, possible_effect, path_1,
                                 path_2)

    def train(self, sentences):
        # Pass 1: extract connective lemmas.
        for sentence in sentences:
            for instance in sentence.causation_instances:
                self._connectives.add(
                    tuple(t.lemma for t in instance.connective))

        # Pass 2: Represent causations as tuples of token indices for easy
        # checking of whether a given possible causation is in the gold
        # standard. List of lists of token indices (1 list per sentence).
        true_causation_tuples = [ 
            [self._get_causation_tuple(i.connective, sentence.get_head(i.cause),
                                       sentence.get_head(i.effect))
             # Filter to pairwise.
             for i in sentence.causation_instances if i.cause and i.effect]
            for sentence in sentences]

        # Pass 3: Gather statistics on connectives/argument parse paths.
        def training_callback(
            sentence_num, connective_lemmas, connective_tokens, possible_cause,
            possible_effect, path_1, path_2):
            causation_tuple = self._get_causation_tuple(
                connective_tokens, possible_cause,
                possible_effect)
            is_causal = causation_tuple in true_causation_tuples[sentence_num]
            # Increment appropriate counter by 1
            key = (connective_lemmas, path_1, path_2)
            if is_causal:
                logging.debug('Logging True for {}'.format(key))
            self._connective_relation_counts[key][is_causal] += 1
        self._operate_on_sentences(sentences, training_callback)

    def test(self, sentences):
        def test_callback(
            sentence_num, connective_lemmas, connective_tokens, possible_cause,
            possible_effect, path_1, path_2):
            key = (connective_lemmas, path_1, path_2)
            counts = self._connective_relation_counts.get(key, None)
            if counts is None:
                # Not seen in training; assumed to be non-causal
                return
            if counts[1] > counts[0]: # go with majority class
                getattr(sentences[sentence_num], self._save_results_in).append(
                    PossibleCausation(
                        None, connective=connective_tokens,
                        cause=[possible_cause], effect=[possible_effect]))
        for sentence in sentences:
            setattr(sentence, self._save_results_in, [])
        self._operate_on_sentences(sentences, test_callback)


class BaselineStage(Stage):
    def __init__(self, name, record_results_in='causation_instances'):
        super(BaselineStage, self).__init__(
            name=name, models=BaselineModel(record_results_in))
        self.produced_attributes = [record_results_in]

    def _extract_parts(self, sentence, is_train):
        return [sentence]

    def _make_evaluator(self):
        return IAAEvaluator(False, False, False, True, True,
                            self.produced_attributes[0])
