from collections import defaultdict
from gflags import DEFINE_list, DEFINE_integer, DEFINE_bool, FLAGS, DuplicateFlagError
import itertools
import logging
from nltk.corpus import wordnet
import numpy as np
from scipy.spatial import distance

from causality_pipelines import IAAEvaluator
from data import Token, ParsedSentence
from pipeline import Stage
from pipeline.models import ClassifierModel, ClassifierPart
from pipeline.feature_extractors import KnownValuesFeatureExtractor, FeatureExtractor, SetValuedFeatureExtractor, \
    VectorValuedFeatureExtractor
from nlp.senna import SennaEmbeddings

try:
    DEFINE_list(
        'tregex_cc_features',
        'cause_pos,effect_pos,wordsbtw,deppath,deplen,connective,cn_lemmas,'
        'tenses,cause_case_children,effect_case_children,domination,'
        'vector_dist,vector_cos_dist'.split(','),
        'Features to use for TRegex-based classifier model')
    DEFINE_integer('tregex_cc_max_wordsbtw', 10,
                   "Maximum number of words between phrases before just making"
                   " the value the max")
    DEFINE_integer('tregex_cc_max_dep_path_len', 3,
                   "Maximum number of dependency path steps to allow before"
                   " just making the value 'LONG-RANGE'")
    DEFINE_bool('tregex_cc_print_test_instances', False,
                'Whether to print differing IAA results during evaluation')
    DEFINE_bool('tregex_cc_tuple_correctness', False,
                'Whether a candidate instance should be considered correct in'
                ' training based on (connective, cause, effect), as opposed to'
                ' just connectives.')
except DuplicateFlagError as e:
    logging.warn('Ignoring redefinition of flag %s' % e.flagname)


class TRegexClassifierPart(ClassifierPart):
    def __init__(self, possible_causation):
        self.sentence = possible_causation.sentence
        self.cause = possible_causation.cause
        self.effect = possible_causation.effect
        self.cause_head = self.sentence.get_head(possible_causation.cause)
        self.effect_head = self.sentence.get_head(possible_causation.effect)
        self.connective = possible_causation.connective
        self.connective_head = self.sentence.get_head(self.connective)
        # TODO: Update this once we have multiple pattern matches sorted out.
        self.connective_pattern = possible_causation.matching_patterns[0]

        if possible_causation.true_causation_instance:
            if FLAGS.tregex_cc_tuple_correctness:
                true_cause_head = self.sentence.get_head(
                    possible_causation.true_causation_instance.cause)
                true_effect_head = self.sentence.get_head(
                    possible_causation.true_causation_instance.effect)
                label = (true_cause_head is self.cause_head and
                         true_effect_head is self.effect_head)
            else:
                label = True
        else:
            label = False
        super(TRegexClassifierPart, self).__init__(self.sentence, label)


class TRegexClassifierModel(ClassifierModel):
    def __init__(self, classifier):
        super(TRegexClassifierModel, self).__init__(
            TRegexClassifierPart,
            TRegexClassifierModel.FEATURE_EXTRACTORS,
            FLAGS.tregex_cc_features, classifier)

    @staticmethod
    def words_btw_heads(part):
        words_btw = part.instance.count_words_between(
            part.cause_head, part.effect_head)
        return min(words_btw, FLAGS.tregex_cc_max_wordsbtw)

    @staticmethod
    def extract_dep_path(part):
        deps = part.instance.extract_dependency_path(
            part.cause_head, part.effect_head, False)
        if len(deps) > FLAGS.tregex_cc_max_dep_path_len:
            return 'LONG-RANGE'
        else:
            return str(deps)

    # We're going to be extracting tenses for pairs of heads for the same
    # sentence. That means we'll get calls for the same head repeatedly, so we
    # cache them for as long as we're dealing with the same sentence.
    # TODO: Make framework send "done training" or "done testing" signals to
    # tell classifier to clear caches.
    __cached_tenses = {}
    __cached_tenses_sentence = None
    @staticmethod
    def extract_tense(head):
        if head.parent_sentence is (
            TRegexClassifierModel.__cached_tenses_sentence):
            try:
                return TRegexClassifierModel.__cached_tenses[head]
            except KeyError:
                pass
        else:
            TRegexClassifierModel.__cached_tenses_sentence = (
                head.parent_sentence)
            TRegexClassifierModel.__cached_tenses = {}

        tense = head.parent_sentence.get_auxiliaries_string(head)
        TRegexClassifierModel.__cached_tenses[head] = tense
        return tense

    @staticmethod
    def extract_daughter_deps(part):
        sentence = part.sentence
        deps = sentence.get_children(part.connective_head)
        edge_labels = [label for label, _ in deps]
        edge_labels.sort()
        return tuple(edge_labels)

    @staticmethod
    def extract_incoming_dep(part):
        edge_label, _parent = part.sentence.get_most_direct_parent(
            part.connective_head)
        return edge_label

    @staticmethod
    def get_verb_children_deps(part):
        if part.connective_head.pos not in Token.VERB_TAGS:
            return 'Non-verb'

        sentence = part.sentence
        children = [child for _, child in
                    sentence.get_children(part.connective_head)]
        verb_children_deps = set()
        for child in children:
            child_deps = [dep for dep, _ in sentence.get_children(child)]
            verb_children_deps.update(child_deps)

        return tuple(verb_children_deps)

    @staticmethod
    def extract_parent_pos(part):
        return part.sentence.get_most_direct_parent(part.connective_head)[1].pos

    _ALL_POS_PAIRS = ['/'.join(tags) for tags in itertools.product(
                        Token.ALL_POS_TAGS, Token.ALL_POS_TAGS)]
    @staticmethod
    def extract_pos_bigram(part, argument_head):
        if argument_head.index < 2:
            prev_pos = 'NONE'
        else:
            previous_token = part.sentence.tokens[argument_head.index - 1]
            # TODO: would this be helpful or harmful?
            # while previous_token.pos in Token.PUNCT_TAGS:
            #     previous_token = part.sentence.tokens[
            #         previous_token.index - 1]
            prev_pos = previous_token.pos
        '/'.join([prev_pos, argument_head.pos])

    @staticmethod
    def extract_wn_hypernyms(token):
        ''' Extracts all Wordnet hypernyms, including the token's lemma. '''
        wn_pos_key = token.get_gen_pos()[0].lower()
        if wn_pos_key == 'j': # correct adjective tag for Wordnet
            wn_pos_key = 'a'
        try:
            synsets = wordnet.synsets(token.lemma, pos=wn_pos_key)
        except KeyError: # Invalid POS tag
            return []
        
        synsets_with_hypernyms = set()
        for synset in synsets:
            for hypernym_path in synset.hypernym_paths():
                synsets_with_hypernyms.update(hypernym_path)

        return tuple(synset.name() for synset in synsets_with_hypernyms)

    @staticmethod
    def extract_case_children(arg_head):
        child_tokens = arg_head.parent_sentence.get_children(arg_head, 'case')
        child_tokens.sort(key=lambda token: token.index)
        return ' '.join([token.lemma for token in child_tokens])

    _embeddings = None # only initialize if being used
    @staticmethod
    def extract_vector(arg_head):
        if not TRegexClassifierModel._embeddings:
            TRegexClassifierModel._embeddings = SennaEmbeddings()
        try:
            return TRegexClassifierModel._embeddings[arg_head.original_text]
        except KeyError: # Unknown word; return special vector
            return TRegexClassifierModel._embeddings['UNKNOWN']

    @staticmethod
    def extract_vector_dist(head1, head2):
        v1 = TRegexClassifierModel.extract_vector(head1)
        v2 = TRegexClassifierModel.extract_vector(head2)
        return np.linalg.norm(v1 - v2)

    @staticmethod
    def extract_vector_cos_dist(head1, head2):
        v1 = TRegexClassifierModel.extract_vector(head1)
        v2 = TRegexClassifierModel.extract_vector(head2)
        return distance.cosine(v1, v2)

    # We can't initialize this properly yet because we don't have access to the
    # class' static methods to define the list.
    FEATURE_EXTRACTORS = []


TRegexClassifierModel.FEATURE_EXTRACTORS = [
    KnownValuesFeatureExtractor('cause_pos', lambda part: part.cause_head.pos,
                                Token.ALL_POS_TAGS),
    KnownValuesFeatureExtractor('effect_pos', lambda part: part.effect_head.pos,
                                Token.ALL_POS_TAGS),
    KnownValuesFeatureExtractor('pos_pair', lambda part: '/'.join([
                                    part.cause_head.pos, part.effect_head.pos]),
                                TRegexClassifierModel._ALL_POS_PAIRS),
    KnownValuesFeatureExtractor(
        'cause_pos_bigram',
        lambda part: TRegexClassifierModel.extract_pos_bigram(part,
                                                              part.cause_head),
                                TRegexClassifierModel._ALL_POS_PAIRS),
    KnownValuesFeatureExtractor(
        'effect_pos_bigram',
        lambda part: TRegexClassifierModel.extract_pos_bigram(part,
                                                              part.effect_head),
                                TRegexClassifierModel._ALL_POS_PAIRS),
    # Generalized POS tags don't seem to be that useful.
    KnownValuesFeatureExtractor(
        'cause_pos_gen', lambda part: part.cause_head.get_gen_pos(),
        Token.ALL_POS_TAGS),
    KnownValuesFeatureExtractor(
        'effect_pos_gen', lambda part: part.effect_head.get_gen_pos(),
        Token.ALL_POS_TAGS),
    FeatureExtractor('wordsbtw', TRegexClassifierModel.words_btw_heads,
                     FeatureExtractor.FeatureTypes.Numerical),
    FeatureExtractor('deppath', TRegexClassifierModel.extract_dep_path),
    FeatureExtractor('deplen',
                     lambda part: len(part.instance.extract_dependency_path(
                        part.cause_head, part.effect_head)),
                     FeatureExtractor.FeatureTypes.Numerical),
    # TODO: Update this once we have multiple pattern matches sorted out.
    FeatureExtractor('connective', lambda part: part.connective_pattern),
    FeatureExtractor('tenses',
                     lambda part: '/'.join(
                        [TRegexClassifierModel.extract_tense(head)
                         for head in part.cause_head, part.effect_head])),
    FeatureExtractor('cn_daughter_deps',
                     TRegexClassifierModel.extract_daughter_deps),
    FeatureExtractor('cn_incoming_dep',
                     TRegexClassifierModel.extract_incoming_dep),
    FeatureExtractor('verb_children_deps',
                     TRegexClassifierModel.get_verb_children_deps),
    FeatureExtractor('cn_parent_pos', TRegexClassifierModel.extract_parent_pos),
    FeatureExtractor('cn_words',
                     lambda part: ' '.join([t.original_text
                                            for t in part.connective])),
    FeatureExtractor('cn_lemmas',
                     lambda part: ' '.join([t.lemma
                                            for t in part.connective])),
    SetValuedFeatureExtractor(
        'cause_hypernyms',
        lambda part: TRegexClassifierModel.extract_wn_hypernyms(
            part.cause_head)),
    SetValuedFeatureExtractor(
        'effect_hypernyms',
        lambda part: TRegexClassifierModel.extract_wn_hypernyms(
            part.effect_head)),
    FeatureExtractor('cause_case_children',
                     lambda part: TRegexClassifierModel.extract_case_children(
                         part.cause_head)),
    FeatureExtractor('effect_case_children',
                     lambda part: TRegexClassifierModel.extract_case_children(
                         part.effect_head)),
    KnownValuesFeatureExtractor('domination',
        lambda part: part.sentence.get_domination_relation(
        part.cause_head, part.effect_head),
        range(len(ParsedSentence.DOMINATION_DIRECTION))),
    VectorValuedFeatureExtractor(
        'cause_vector',
        lambda part: TRegexClassifierModel.extract_vector(part.cause_head)),
    VectorValuedFeatureExtractor(
        'effect_vector',
        lambda part: TRegexClassifierModel.extract_vector(part.cause_head)),
    FeatureExtractor('vector_dist',
                     lambda part: TRegexClassifierModel.extract_vector_dist(
                        part.cause_head, part.effect_head),
                     FeatureExtractor.FeatureTypes.Numerical),
    FeatureExtractor('vector_cos_dist',
                     lambda part: TRegexClassifierModel.extract_vector_cos_dist(
                        part.cause_head, part.effect_head),
                     FeatureExtractor.FeatureTypes.Numerical),
]


class TRegexClassifierStage(Stage):
    def __init__(self, classifier, name):
        super(TRegexClassifierStage, self).__init__(
            name=name, models=TRegexClassifierModel(classifier))

    consumed_attributes = ['possible_causations']

    def _extract_parts(self, sentence, is_train):
        return [TRegexClassifierPart(p) for p in sentence.possible_causations]

    def _decode_labeled_parts(self, sentence, labeled_parts):
        # Deduplicate the results.

        tokens_to_parts = defaultdict(int)
        positive_parts = [p for p in labeled_parts if p.label]
        for part in positive_parts:
            # Count every instance each connective word is part of.
            for connective_token in part.connective:
                tokens_to_parts[connective_token] += 1

        sentence.causation_instances = []
        for part in positive_parts:
            keep_part = True
            for token in part.connective:
                if tokens_to_parts[token] > 1:
                    # Assume that if there are other matches for a word, and
                    # this match relies on Steiner nodes, it's probably wrong.
                    # TODO: should we worry about cases where all connectives
                    # on this word were found using Steiner patterns?
                    if 'steiner_0' in part.connective_pattern:
                        keep_part = False
                        break
            if keep_part:
                sentence.add_causation_instance(
                    connective=part.connective,
                    cause=part.cause, effect=part.effect)

    def _make_evaluator(self):
        # TODO: provide both pairwise and non-pairwise stats
        return IAAEvaluator(False, False, FLAGS.tregex_cc_print_test_instances,
                            True, True)
