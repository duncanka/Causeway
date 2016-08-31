from collections import Counter, defaultdict, OrderedDict
from gflags import (DEFINE_list, DEFINE_integer, DEFINE_bool, FLAGS,
                    DuplicateFlagError, DEFINE_float, DEFINE_enum)
from itertools import chain, product
import logging
import math
from nltk.corpus import wordnet
from nltk.util import skipgrams
import numpy as np
from scipy.spatial import distance
import sklearn
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.pipeline import Pipeline as SKLPipeline

from causality_pipelines import (IAAEvaluator, StanfordNERStage,
                                 RELATIVE_POSITIONS)
from data import Token, StanfordParsedSentence, CausationInstance
from iaa import make_annotation_comparator, stringify_connective
from nlp.senna import SennaEmbeddings
from pipeline import Stage
from pipeline.featurization import (
    KnownValuesFeatureExtractor, FeatureExtractor, SetValuedFeatureExtractor,
    VectorValuedFeatureExtractor, NestedFeatureExtractor,
    MultiNumericalFeatureExtractor, ThresholdedFeatureExtractor, Featurizer)
from pipeline.models.structured import StructuredDecoder, StructuredModel
from skpipeline import (make_featurizing_estimator,
                        make_mostfreq_featurizing_estimator)
from util import powerset
from util.diff import SequenceDiff
from util.scipy import (AutoWeightedVotingClassifier, make_logistic_score,
                        prob_sum_score)
from util.metrics import ClassificationMetrics, diff_binary_vectors
from nltk.metrics.scores import accuracy


try:
    DEFINE_list(
        'filter_features',
        'cn_words,cause_hypernyms,effect_hypernyms,cause_vector,effect_vector,'
        'cause_pos,effect_pos,cause_pos_gen,effect_pos_gen,cause_tense,'
        'effect_tense,cn_incoming_dep,cause_daughter_deps,effect_daughter_deps,'
        'verb_children_deps,cn_parent_pos,all_effect_closed_children,'
        'all_cause_closed_children,heads_rel_pos,cause_closed_children,'
        'effect_closed_children,commas_btw,args_rel_pos,deplen,'
        'all_cause_closed_children_deps,cause_closed_children_deps,'
        'all_effect_closed_children_deps,effect_closed_children_deps,'
        'cause_prep_start,effect_prep_start,cause_pos_skipgrams,'
        'effect_pos_skipgrams,cause_lemma_skipgrams,effect_lemma_skipgrams,'
        'cause_tense:effect_tense,cause_ner:effect_ner'.split(','),
        'Features to use for pattern-based candidate classifier model')
    DEFINE_list('filter_features_to_cancel', [],
                'Features from the features list to cancel (useful with'
                ' default features list)')
    DEFINE_integer('filter_max_wordsbtw', 10,
                   "Maximum number of words between phrases before just making"
                   " the value the max")
    DEFINE_integer('filter_max_dep_path_len', 4,
                   "Maximum number of dependency path steps to allow before"
                   " just making the value 'LONG-RANGE'")
    DEFINE_bool('filter_print_test_instances', False,
                'Whether to print differing IAA results during evaluation')
    DEFINE_bool('filter_diff_correctness', None,
                'Whether a candidate instance should be considered correct in'
                ' training based on diffing the sequence of true instances and'
                ' the sequence of proposed instances. If False, then any'
                ' proposed instance that has been matched to a true instance'
                " will be marked as true, even if it's a duplicate. Default is"
                " True for regex pipeline and False for TRegex.")
    DEFINE_bool('filter_train_with_partials', False,
                'Whether to train the candidate classifier model counting'
                ' partial overlap as correct')
    DEFINE_integer('filter_feature_select_k', -1,
                   'Specifies how many features to keep in feature selection'
                   ' for per-connective causality filters. -1 means no feature'
                   ' selection.')
    DEFINE_float('filter_prob_cutoff', 0.45,
                 'Probability threshold for instances to mark as causal',
                 0.0, 1.0)
    DEFINE_bool('filter_record_raw_accuracy', True,
                'Whether to include raw classification accuracy in the'
                ' evaluation scores for the causation filter')
    DEFINE_bool('filter_scale_C', False,
                'Whether to scale the regularization strength on the filter'
                ' classifier')
    DEFINE_bool('filter_save_scored', False,
                'Whether to save the classifier probabilities and true labels')
    DEFINE_integer('filter_sg_lemma_threshold', 4,
                   'Minimum # of arguments that a lemma skipgram must appear in'
                   ' for it to be considered as a skipgram feature')
    DEFINE_float('filter_tuning_pct', 0.5,
                 'Fraction of training data to devote to tuning classifier'
                 ' weights instead of training per-connective classifiers'
                 ' (when auto-weighting is enabled)',
                 0.0, 1.0)
    DEFINE_float('filter_wt_score_slope', None,
                 'Slope parameter for the logistic function used in weighting'
                 ' classifiers. If None, no logistic function is used; the'
                 ' probabilities of the correct answers are summed directly.')
    DEFINE_enum('filter_classifiers', 'global,mostfreq,perconn',
                 [','.join(sorted(x)) # alphabetize
                  for x in powerset(['global', 'perconn', 'mostfreq']) if x],
                'Which classifiers should be trained and allowed to vote for'
                ' each connective type')
    DEFINE_bool('filter_auto_weight', False,
                'Whether to automatically weight classifiers for each'
                ' connective')
    DEFINE_bool('filter_global_for_all_same', True,
                'Whether to include the global classifier for connectives where'
                ' all training instances have the same label. If False, only'
                ' the mostfreq classifier is used.')
    DEFINE_bool('filter_record_feature_weights', False,
                'Whether to record the range of classifier feature weights for'
                ' each feature type') # only works for LR classifier
except DuplicateFlagError as e:
    logging.warn('Ignoring redefinition of flag %s' % e.flagname)


class PatternFilterPart(object):
    def __init__(self, possible_causation, connective_correct=None):
        self.possible_causation = possible_causation
        self.sentence = possible_causation.sentence
        self.cause = possible_causation.cause
        self.effect = possible_causation.effect
        # If we're missing an argument, we'll want to keep the instance around
        # for majority-class classification, but it'll be excluded if we build a
        # proper classifier.
        if possible_causation.cause and possible_causation.effect:
            self.cause_head = self.sentence.get_head(possible_causation.cause)
            self.effect_head = self.sentence.get_head(possible_causation.effect)
        else:
            self.cause_head = None
            self.effect_head = None
        self.connective = possible_causation.connective
        self.connective_head = self.sentence.get_head(self.connective)
        self.connective_patterns = possible_causation.matching_patterns
        self.connective_correct = connective_correct


class CausalClassifierModel(object):
    @staticmethod
    def _get_gold_labels(classifier_parts):
        return [int(part.connective_correct) for part in classifier_parts]

    #############################
    # Feature extraction methods
    #############################

    @staticmethod
    def get_pos_with_copulas(token):
        if token.parent_sentence.is_copula_head(token):
            return token.pos + '<COP>'
        else:
            return token.pos

    @staticmethod
    def words_btw_heads(part):
        words_btw = part.sentence.count_words_between(
            part.cause_head, part.effect_head)
        return min(words_btw, FLAGS.filter_max_wordsbtw)

    @staticmethod
    def extract_dep_path(part):
        deps = part.sentence.extract_dependency_path(
            part.cause_head, part.effect_head, False)
        if len(deps) > FLAGS.filter_max_dep_path_len:
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
            CausalClassifierModel.__cached_tenses_sentence):
            try:
                return CausalClassifierModel.__cached_tenses[head]
            except KeyError:
                pass
        else:
            CausalClassifierModel.__cached_tenses_sentence = (
                head.parent_sentence)
            CausalClassifierModel.__cached_tenses = {}

        tense = head.parent_sentence.get_auxiliaries_string(head)
        CausalClassifierModel.__cached_tenses[head] = tense
        return tense

    @staticmethod
    def extract_daughter_deps(part, head):
        sentence = part.sentence
        deps = sentence.get_children(head)
        edge_labels = [label for label, _ in deps]
        edge_labels.sort()
        return tuple(edge_labels)

    @staticmethod
    def extract_cn_daughter_deps(part):
        return CausalClassifierModel.extract_daughter_deps(part,
                                                           part.connective_head)

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

        return ','.join(sorted(verb_children_deps))

    @staticmethod
    def extract_parent_pos(part):
        _edge_label, parent = part.sentence.get_most_direct_parent(
            part.connective_head)
        if parent is None:
            return None
        return CausalClassifierModel.get_pos_with_copulas(parent)

    _ALL_POS_PAIRS = ['/'.join(tags) for tags in product(
                        Token.ALL_POS_TAGS, Token.ALL_POS_TAGS)]
    @staticmethod
    def extract_pos_bigram(part, arg_head):
        if arg_head.index < 2:
            prev_pos = 'NONE'
            pos = CausalClassifierModel.get_pos_with_copulas(arg_head)
        else:
            previous_token = part.sentence.tokens[arg_head.index - 1]
            # TODO: would this be helpful or harmful?
            # while previous_token.pos in Token.PUNCT_TAGS:
            #     previous_token = part.sentence.tokens[
            #         previous_token.index - 1]
            prev_pos, pos = [CausalClassifierModel.get_pos_with_copulas(token)
                             for token in previous_token, arg_head]
        '/'.join([prev_pos, pos])

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
        if not CausalClassifierModel._embeddings:
            CausalClassifierModel._embeddings = SennaEmbeddings()
        try:
            return CausalClassifierModel._embeddings[
                arg_head.lowered_text]
        except KeyError: # Unknown word; return special vector
            return CausalClassifierModel._embeddings['UNKNOWN']

    @staticmethod
    def extract_vector_dist(head1, head2):
        v1 = CausalClassifierModel.extract_vector(head1)
        v2 = CausalClassifierModel.extract_vector(head2)
        return np.linalg.norm(v1 - v2)

    @staticmethod
    def extract_vector_cos_dist(head1, head2):
        v1 = CausalClassifierModel.extract_vector(head1)
        v2 = CausalClassifierModel.extract_vector(head2)
        return distance.cosine(v1, v2)

    @staticmethod
    def count_commas_between(cause, effect):
        if not (StanfordParsedSentence.is_contiguous(cause) and
                StanfordParsedSentence.is_contiguous(effect)):
            return 0 # TODO: something more intelligent?

        if cause[0].index < effect[0].index: # cause comes first
            start, end = cause[-1].index + 1, effect[0].index
        else:
            start, end = effect[-1].index + 1, cause[0].index

        count = 0
        for token in cause[0].parent_sentence.tokens[start:end]:
            if token.original_text == ',':
                count += 1
        return count

    @staticmethod
    def cause_pos_wrt_effect(cause, effect):
        # NOTE: this works only because cause and effect spans are sorted.
        if effect[0].index > cause[-1].index: # effect starts after cause ends
            return RELATIVE_POSITIONS.Before
        elif cause[0].index > effect[-1].index: # cause starts after effect ends
            return RELATIVE_POSITIONS.After
        else:
            return RELATIVE_POSITIONS.Overlapping

    @staticmethod
    def get_pp_preps(argument_head):
        sentence = argument_head.parent_sentence
        children = sentence.get_children(argument_head, 'case')
        return [c.lemma for c in children if c.pos == 'IN']

    @staticmethod
    def is_negated(token):
        sentence = token.parent_sentence
        children = sentence.get_children(token, 'neg')
        return bool(children)

    @staticmethod
    def has_negated_child(argument_head):
        sentence = argument_head.parent_sentence
        children = sentence.get_children(argument_head, '*')
        return any(CausalClassifierModel.is_negated(child)
                   for child in children if child.get_gen_pos() == 'NN')

    @staticmethod
    def is_comp(argument_head):
        sentence = argument_head.parent_sentence
        edge_label, _ = sentence.get_most_direct_parent(argument_head)
        return edge_label == 'ccomp'

    @staticmethod
    def starts_w_comp(argument_span):
        first_token = argument_span[0]
        return first_token.lemma in ['that', 'for', 'to']

    @staticmethod
    def initial_prep(argument_span):
        first_token = argument_span[0]
        if first_token.pos == 'IN':
            return first_token.lemma
        else:
            return 'None'

    # Adapted from
    # http://mailman.uib.no/public/corpora/2011-November/014318.html.
    # (Omitted: numbers, times, places, non-extrapositional pronouns.)
    ALL_CLOSED_CLASS = set(
        "be to there" # Copulas, infinitives, and extraposition
        " all both some many much more most too enough few little fewer less"
            " least than one" # Quantifiers and comparisons
        " this these that those a the any each" # Other determiners
        " no not nobody nothing none nowhere never" # Negations
        " yes everywhere always" # Affirmations
        " have can cannot could shall should may might must do will"
            " would" # Modals
        " albeit although because 'cause if neither since so than that though"
            " lest 'til till unless until whereas whether while" # Subordinating
        " & and 'n 'n' but either et neither nor or plus v. versus vs."
            " yet" # Coordinating
        " ago as at besides between by except for from in into like"
            " notwithstanding of off on onto out over per since through"
            " throughout"
            " unto up upon versus via vs. with within without" # Prepositions
        .split(" "))

    @staticmethod
    def closed_class_children(arg_head):
        child_tokens = arg_head.parent_sentence.get_children(arg_head, '*')
        child_tokens.sort(key=lambda token: token.index)
        return tuple(token.lemma for token in child_tokens
                     if token.lemma in CausalClassifierModel.ALL_CLOSED_CLASS)

    @staticmethod
    def closed_class_children_deps(arg_head):
        child_edges_and_tokens = arg_head.parent_sentence.get_children(arg_head)
        child_edges_and_tokens.sort(key=lambda pair: pair[1].index)
        return tuple('/'.join([token.lemma, edge_label])
                     for edge_label, token in child_edges_and_tokens
                     if token.lemma in CausalClassifierModel.ALL_CLOSED_CLASS)

    @staticmethod
    def get_ner_distance(arg, ner_tag_type):
        for i, token in enumerate(arg):
            if token.ner_tag == ner_tag_type:
                return i
        return -1

    @staticmethod
    def get_pos_skipgrams(arg):
        pos_skipgrams = skipgrams([t.pos for t in arg], 2, 1)
        return Counter(' '.join(skipgram) for skipgram in pos_skipgrams)

    @staticmethod
    def get_lemma_skipgrams(arg):
        lemma_skipgrams = skipgrams([t.lemma for t in arg], 2, 1)
        return Counter(' '.join(skipgram) for skipgram in lemma_skipgrams)

    all_feature_extractors = []
    per_conn_and_shared_feature_extractors = []
    global_and_shared_feature_extractors = []


Numerical = FeatureExtractor.FeatureTypes.Numerical
Binary = FeatureExtractor.FeatureTypes.Binary

CausalClassifierModel.per_connective_feature_extractors = [
    SetValuedFeatureExtractor(
        'connective', lambda part: part.connective_patterns),
    FeatureExtractor('cn_words',
                     lambda part: ' '.join([t.lowered_text
                                            for t in part.connective])),
    FeatureExtractor('cn_lemmas',
                     lambda part: ' '.join([t.lemma
                                            for t in part.connective]))
]

CausalClassifierModel.global_feature_extractors = [
    SetValuedFeatureExtractor(
        'cause_hypernyms',
        lambda part: CausalClassifierModel.extract_wn_hypernyms(
            part.cause_head)),
    SetValuedFeatureExtractor(
        'effect_hypernyms',
        lambda part: CausalClassifierModel.extract_wn_hypernyms(
            part.effect_head)),
    VectorValuedFeatureExtractor(
        'cause_vector',
        lambda part: CausalClassifierModel.extract_vector(
                        part.cause_head)),
    VectorValuedFeatureExtractor(
        'effect_vector',
        lambda part: CausalClassifierModel.extract_vector(
                        part.cause_head)),
    FeatureExtractor(
        'vector_dist',
        lambda part: CausalClassifierModel.extract_vector_dist(
                         part.cause_head, part.effect_head), Numerical),
    FeatureExtractor(
        'vector_cos_dist',
        lambda part: CausalClassifierModel.extract_vector_cos_dist(
                        part.cause_head, part.effect_head), Numerical),
]

CausalClassifierModel.shared_feature_extractors = [
    KnownValuesFeatureExtractor(
        'cause_pos',
        lambda part: CausalClassifierModel.get_pos_with_copulas(
                        part.cause_head),
        Token.ALL_POS_TAGS),
    KnownValuesFeatureExtractor(
        'effect_pos',
        lambda part: CausalClassifierModel.get_pos_with_copulas(
                        part.effect_head),
        Token.ALL_POS_TAGS),
    KnownValuesFeatureExtractor(
        'cause_pos_bigram',
        lambda part: CausalClassifierModel.extract_pos_bigram(
                             part, part.cause_head),
                         CausalClassifierModel._ALL_POS_PAIRS),
    KnownValuesFeatureExtractor(
        'effect_pos_bigram',
        lambda part: CausalClassifierModel.extract_pos_bigram(
                             part, part.effect_head),
                         CausalClassifierModel._ALL_POS_PAIRS),
    # Generalized POS tags don't seem to be that useful.
    KnownValuesFeatureExtractor(
        'cause_pos_gen', lambda part: part.cause_head.get_gen_pos(),
        Token.ALL_POS_TAGS),
    KnownValuesFeatureExtractor(
        'effect_pos_gen', lambda part: part.effect_head.get_gen_pos(),
        Token.ALL_POS_TAGS),
    FeatureExtractor('wordsbtw', CausalClassifierModel.words_btw_heads,
                     Numerical),
    FeatureExtractor('deppath', CausalClassifierModel.extract_dep_path),
    SetValuedFeatureExtractor(
        'deps_on_path',
        lambda part: part.sentence.extract_dependency_path(
            part.cause_head, part.effect_head, False)),
    FeatureExtractor('deplen',
                     lambda part: len(part.sentence.extract_dependency_path(
                        part.cause_head, part.effect_head)), Numerical),
    FeatureExtractor('cause_tense',
                     lambda part: CausalClassifierModel.extract_tense(
                        part.cause_head)),
    FeatureExtractor('effect_tense',
                     lambda part: CausalClassifierModel.extract_tense(
                        part.effect_head)),
    FeatureExtractor('cn_incoming_dep',
                     CausalClassifierModel.extract_incoming_dep),
    SetValuedFeatureExtractor('cn_daughter_deps',
                              CausalClassifierModel.extract_cn_daughter_deps),
    SetValuedFeatureExtractor(
        'cause_daughter_deps',
        lambda part: CausalClassifierModel.extract_daughter_deps(
                         part, part.cause_head)),
    SetValuedFeatureExtractor(
        'effect_daughter_deps',
        lambda part: CausalClassifierModel.extract_daughter_deps(
                         part, part.effect_head)),
    FeatureExtractor('verb_children_deps',
                     CausalClassifierModel.get_verb_children_deps),
    FeatureExtractor('cn_parent_pos',
                     CausalClassifierModel.extract_parent_pos),
    FeatureExtractor('all_cause_closed_children',
                     lambda part: ' '.join(
                         CausalClassifierModel.closed_class_children(
                            part.cause_head))),
    FeatureExtractor('all_effect_closed_children',
                     lambda part: ' '.join(
                         CausalClassifierModel.closed_class_children(
                            part.effect_head))),
    SetValuedFeatureExtractor(
        'cause_closed_children',
        lambda part: CausalClassifierModel.closed_class_children(
                        part.cause_head)),
    SetValuedFeatureExtractor(
        'effect_closed_children',
        lambda part: CausalClassifierModel.closed_class_children(
                        part.effect_head)),
    KnownValuesFeatureExtractor('domination',
        lambda part: part.sentence.get_domination_relation(
        part.cause_head, part.effect_head),
        range(len(StanfordParsedSentence.DOMINATION_DIRECTION))),
    # TODO: remove for global classifier? (Too construction-specific)
    KnownValuesFeatureExtractor(
        'cause_ner',
        lambda part: StanfordNERStage.NER_TYPES[part.cause_head.ner_tag],
        StanfordNERStage.NER_TYPES),
    KnownValuesFeatureExtractor(
        'effect_ner',
        lambda part: StanfordNERStage.NER_TYPES[part.effect_head.ner_tag],
        StanfordNERStage.NER_TYPES),
    FeatureExtractor('commas_btw',
                     lambda part: CausalClassifierModel.count_commas_between(
                                      part.cause, part.effect), Numerical),
    FeatureExtractor('cause_len', lambda part: len(part.cause), Numerical),
    FeatureExtractor('effect_len', lambda part: len(part.effect), Numerical),
    FeatureExtractor('args_rel_pos',
                     lambda part: CausalClassifierModel.cause_pos_wrt_effect(
                                      part.cause, part.effect)),
    FeatureExtractor('heads_rel_pos',
                     lambda part: CausalClassifierModel.cause_pos_wrt_effect(
                                      [part.cause_head], [part.effect_head])),
    FeatureExtractor(
        'all_cause_closed_children_deps',
        lambda part: ' '.join(CausalClassifierModel.closed_class_children_deps(
                        part.cause_head))),
    FeatureExtractor(
        'all_effect_closed_children_deps',
        lambda part: ' '.join(CausalClassifierModel.closed_class_children_deps(
                        part.effect_head))),
    SetValuedFeatureExtractor(
        'cause_closed_children_deps',
        lambda part: CausalClassifierModel.closed_class_children_deps(
                        part.cause_head)),
    SetValuedFeatureExtractor(
        'effect_closed_children_deps',
        lambda part: CausalClassifierModel.closed_class_children_deps(
                        part.effect_head)),
    FeatureExtractor(
        'cause_neg',
        lambda part: CausalClassifierModel.is_negated(part.cause_head),
        Binary),
    FeatureExtractor(
        'effect_neg',
        lambda part: CausalClassifierModel.is_negated(part.effect_head),
        Binary),
    FeatureExtractor(
        'cause_neg_child',
        lambda part: CausalClassifierModel.has_negated_child(
                        part.cause_head),
        Binary),
    FeatureExtractor(
        'effect_neg_child',
        lambda part: CausalClassifierModel.has_negated_child(
                        part.effect_head),
        Binary),
    # TODO: look for other negative words
    FeatureExtractor(
        'cause_comp',
        lambda part: CausalClassifierModel.is_comp(part.cause_head),
        Binary),
    FeatureExtractor(
        'effect_comp',
        lambda part: CausalClassifierModel.is_comp(part.effect_head),
        Binary),
    FeatureExtractor(
        'cause_comp_start',
        lambda part: CausalClassifierModel.starts_w_comp(part.cause),
        Binary),
    FeatureExtractor(
        'effect_comp_start',
        lambda part: CausalClassifierModel.starts_w_comp(part.effect),
        Binary),
    FeatureExtractor(
        'cause_prep_start',
        lambda part: CausalClassifierModel.initial_prep(part.cause)),
    FeatureExtractor(
        'effect_prep_start',
        lambda part: CausalClassifierModel.initial_prep(part.effect)),
    NestedFeatureExtractor(
        'cause_ner_distances',
        [FeatureExtractor(tag_name,
                          lambda part: CausalClassifierModel.get_ner_distance(
                                           part.cause, tag_type))
         for tag_type, tag_name in enumerate(StanfordNERStage.NER_TYPES)]),
    NestedFeatureExtractor(
        'effect_ner_distances',
        [FeatureExtractor(tag_name,
                          lambda part: CausalClassifierModel.get_ner_distance(
                                           part.effect, tag_type))
         for tag_type, tag_name in enumerate(StanfordNERStage.NER_TYPES)]),
    MultiNumericalFeatureExtractor(
        'cause_pos_skipgrams',
        lambda part: CausalClassifierModel.get_pos_skipgrams(part.cause)),
    MultiNumericalFeatureExtractor(
        'effect_pos_skipgrams',
        lambda part: CausalClassifierModel.get_pos_skipgrams(part.effect)),
    ThresholdedFeatureExtractor(
        MultiNumericalFeatureExtractor(
            'cause_lemma_skipgrams',
            lambda part: CausalClassifierModel.get_lemma_skipgrams(part.cause)),
        FLAGS.filter_sg_lemma_threshold),
    ThresholdedFeatureExtractor(
        MultiNumericalFeatureExtractor(
            'effect_lemma_skipgrams',
            lambda part: CausalClassifierModel.get_lemma_skipgrams(
                             part.effect)),
        FLAGS.filter_sg_lemma_threshold),
]

CausalClassifierModel.per_conn_and_shared_feature_extractors = (
    CausalClassifierModel.per_connective_feature_extractors
    + CausalClassifierModel.shared_feature_extractors)

CausalClassifierModel.global_and_shared_feature_extractors = (
    CausalClassifierModel.global_feature_extractors
    + CausalClassifierModel.shared_feature_extractors)

CausalClassifierModel.all_feature_extractors = (
    CausalClassifierModel.per_connective_feature_extractors
    + CausalClassifierModel.global_feature_extractors
    + CausalClassifierModel.shared_feature_extractors)


class PatternBasedCausationFilter(StructuredModel):
    def __init__(self, classifier, labels_for_eval, gold_labels_for_eval):
        super(PatternBasedCausationFilter, self).__init__(
            PatternBasedFilterDecoder(labels_for_eval, gold_labels_for_eval,
                                      FLAGS.filter_save_scored))

        self.soft_voting = hasattr(classifier, 'predict_proba')
        self.base_mostfreq_classifier = make_mostfreq_featurizing_estimator(
            'most_freq_classifier')

        selected_features = (set(FLAGS.filter_features)
                             - set(FLAGS.filter_features_to_cancel))

        Featurizer.check_selected_features_list(
            selected_features, CausalClassifierModel.all_feature_extractors)
        if FLAGS.filter_feature_select_k == -1:
            base_per_conn_classifier = sklearn.clone(classifier)
            global_classifier = sklearn.clone(classifier)
        else:
            k_best = SelectKBest(chi2, FLAGS.filter_feature_select_k)
            base_per_conn_classifier = SKLPipeline([
                ('feature_selection', k_best),
                ('classification', sklearn.clone(classifier))
            ])
            global_classifier = sklearn.clone(base_per_conn_classifier)

        per_conn_selected = Featurizer.selected_features_for_featurizer(
            selected_features,
            CausalClassifierModel.per_conn_and_shared_feature_extractors)
        self.base_per_conn_classifier = make_featurizing_estimator(
            base_per_conn_classifier,
            'causality_pipelines.candidate_filter.CausalClassifierModel'
            '.per_conn_and_shared_feature_extractors',
            per_conn_selected, 'perconn_classifier')

        global_selected = Featurizer.selected_features_for_featurizer(
            selected_features,
            CausalClassifierModel.global_and_shared_feature_extractors)
        self.global_classifier = make_featurizing_estimator(
            global_classifier,
            'causality_pipelines.candidate_filter.CausalClassifierModel'
            '.global_and_shared_feature_extractors', global_selected,
            'global_causality_classifier')

        self.classifiers = {}
        comparator = make_annotation_comparator(
            FLAGS.filter_train_with_partials)
        # Comparator for matching CausationInstances against PossibleCausations
        self.connective_comparator = lambda inst1, inst2: comparator(
                                        inst1.connective, inst2.connective)
        # By default, regex, not tregex, should use diff correctness.
        if FLAGS.filter_diff_correctness is None:
            FLAGS.filter_diff_correctness = (
                'tregex' not in FLAGS.pipeline_type)
            logging.debug("Set flag filter_diff_correctness to %s"
                          % FLAGS.filter_diff_correctness)

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['connective_comparator']
        return state

    def _make_parts(self, sentence, is_train):
        if is_train:
            if FLAGS.filter_diff_correctness:
                # In training, we need to match the causation instances the
                # pipeline has thus far detected against the original causation
                # instances (provided by previous pipeline stages). We do this
                # the same way that the IAA code does it internally: by running
                # a diff on the connectives. Except we cheat a bit, and compare
                # PossibleCausations against real CausationInstances.
                parts = []
                # We want the diff to sort by connective position.
                sort_by_key = lambda inst: inst.connective[0].start_offset
                connectives_diff = SequenceDiff(
                    sentence.possible_causations, sentence.causation_instances,
                    self.connective_comparator, sort_by_key)
                for correct_pc, _ in connectives_diff.get_matching_pairs():
                    parts.append(PatternFilterPart(correct_pc, True))
                for incorrect_pc in connectives_diff.get_a_only_elements():
                    parts.append(PatternFilterPart(incorrect_pc, False))
            else:
                parts = [PatternFilterPart(pc, bool(pc.true_causation_instance))
                         for pc in sentence.possible_causations]
            return [pc for pc in parts if pc.cause and pc.effect]
        else:
            # If we're not in training, the initial label doesn't really matter.
            # We do want to filter to only 2-arg matches.
            return [PatternFilterPart(pc, False) for pc in
                    sentence.possible_causations if pc.cause and pc.effect]

    def reset(self):
        super(PatternBasedCausationFilter, self).reset()
        self.classifiers = {}

    @staticmethod
    def _fit_allowing_feature_selection(classifier, data, labels):
        try:
            classifier.fit(data, labels)
        except ValueError: # feature selection failed b/c too few features
            classification_pipeline = classifier.steps[1][1]
            feature_selector = classification_pipeline.steps[0][1]
            feature_selector.k = 'all'
            classifier.fit(data, labels)

    def _get_estimators_for_connective(self, pcs, labels, use_global,
                                       use_per_conn, use_mostfreq, auto_weight):
        if auto_weight:
            # Don't use all the training data when training the per-connective
            # classifier. Instead, reserve some of it for tuning classifier
            # weights.
            num_per_conn_training = int(math.ceil(len(pcs)
                                                  * FLAGS.filter_tuning_pct))
            per_conn_training = pcs[:num_per_conn_training]
            per_conn_training_labels = labels[:num_per_conn_training]
        else:
            # We won't be auto-weighting, so no need to reserve any tuning data.
            num_per_conn_training = len(pcs)
            per_conn_training = pcs
            per_conn_training_labels = labels

        # Some estimators don't deal well with all labels being the same.
        # If this is the case and we're doing per-connective classifiers, it
        # should just default to majority-class anyway, so just do that.
        unique_labels = set(per_conn_training_labels)
        all_same_class = len(unique_labels) < 2
        if all_same_class:
            use_global = use_global and FLAGS.filter_global_for_all_same
            use_per_conn = False
            use_mostfreq = True

        estimators = [('global', self.global_classifier)] if use_global else []

        if use_per_conn:
            per_conn = sklearn.clone(self.base_per_conn_classifier)
            if FLAGS.filter_scale_C:
                per_conn.named_steps['perconn_classifier'].C = math.sqrt(
                    num_per_conn_training / 5.0)
            self._fit_allowing_feature_selection(per_conn, per_conn_training,
                                                 per_conn_training_labels)
            estimators.append(('perconn', per_conn))

        if use_mostfreq:
            mostfreq = sklearn.clone(self.base_mostfreq_classifier)
            mostfreq.fit(per_conn_training, per_conn_training_labels)
            if all_same_class:
                # Manually inform most-frequent classifier of other class.
                mf_classifier = mostfreq.steps[-1][-1]
                mf_classifier.n_classes_ = 2
                class_label = mf_classifier.classes_[0]
                mf_classifier.classes_ = np.array(
                    [False, True], dtype=mf_classifier.classes_.dtype)
                mf_classifier.class_prior_ = np.array(
                    [class_label == False, class_label == True],
                    dtype=mf_classifier.class_prior_.dtype)
            estimators.append(('mostfreq', mostfreq))

        return estimators

    def _train_structured(self, sentences, parts_by_sentence):
        classifier_types = FLAGS.filter_classifiers.split(',')
        auto_weight = FLAGS.filter_auto_weight
        use_global, use_per_conn, use_mostfreq = [
            t in classifier_types for t in ['global', 'perconn', 'mostfreq']]

        all_pcs = [pc for pc in chain.from_iterable(parts_by_sentence)
                   if pc.cause and pc.effect] # train only on 2-arg instances

        if use_global:
            all_labels = CausalClassifierModel._get_gold_labels(all_pcs)
            self._fit_allowing_feature_selection(self.global_classifier,
                                                 all_pcs, all_labels)

        pcs_by_connective = defaultdict(list)
        for pc in all_pcs:
            connective = stringify_connective(pc)
            pcs_by_connective[connective].append(pc)

        if self.soft_voting:
            if FLAGS.filter_wt_score_slope is None:
                score_fn = prob_sum_score
            else:
                score_fn = make_logistic_score(1.0, FLAGS.filter_wt_score_slope,
                                               0.5)
        else:
            score_fn = accuracy

        for connective, pcs in pcs_by_connective.iteritems():
            labels = CausalClassifierModel._get_gold_labels(pcs)
            estimators = self._get_estimators_for_connective(
                pcs, labels, use_global, use_per_conn, use_mostfreq,
                auto_weight)
            classifier = AutoWeightedVotingClassifier(
                estimators=estimators, autofit_weights=auto_weight,
                voting='soft' if self.soft_voting else 'hard',
                score_probas=self.soft_voting, score_fn=score_fn)
            if auto_weight:
                classifier.fit_weights(pcs, labels) # use train + dev for tuning
            else: # weight evenly
                classifier.weights = [1.0 / len(estimators)] * len(estimators)

            self.classifiers[connective] = classifier

        if FLAGS.filter_record_feature_weights:
            self._record_feature_weights()

    def _record_feature_weights(self):
        if not hasattr(self, 'feature_weights'):
            self.feature_weights = defaultdict(list)

        feature_sep = FLAGS.conjoined_feature_sep

        lr_pipelines = [self.global_classifier]
        for classifier in self.classifiers.values():
            try:
                lr_pipelines.append(dict(classifier.estimators)['perconn'])
            except KeyError: # no 'perconn' classifier
                continue

        for lr_pipeline in lr_pipelines:
            updates_by_extractor = defaultdict(list)
            feature_weights = get_weights_for_lr_classifier(lr_pipeline, False)
            for feature_name, weight in feature_weights.iteritems():
                conjoined_names = feature_name.split(feature_sep)
                extractor_name = feature_sep.join([name.split('=')[0]
                                                   for name in conjoined_names])
                updates_by_extractor[extractor_name].append(abs(weight))

            for extractor_name, updates in updates_by_extractor.iteritems():
                extractor_weights = self.feature_weights[extractor_name]
                extractor_weights.extend(updates)

    def _score_parts(self, sentence, possible_causations):
        using_global = 'global' in FLAGS.filter_classifiers.split(',')
        if self.soft_voting:
            scores = []
            for pc in possible_causations:
                try:
                    classifier = self.classifiers[stringify_connective(pc)]
                    true_class_index = classifier.le_.transform(1)
                    try:
                        pc_scores = [classifier.predict_proba(
                                         [pc])[0, true_class_index]]
                    except ZeroDivisionError: # happens if all scores are 0
                        pc_scores = [0]
                    # TODO: Make this add NaNs to the right places depending on
                    # which estimators are present.
                    pc_scores += [c.predict_proba([pc])[0, true_class_index]
                                  for c in classifier.estimators_]
                except KeyError:
                    # We didn't encounter any 2-argument instances of this
                    # pattern in training, so we have no classifier for it.
                    connective_text = ' '.join(t.lemma for t in pc.connective)
                    if using_global:
                        true_class_index = np.where(
                            self.global_classifier.classes_ == True)[0][0]
                        global_score = self.global_classifier.predict_proba(
                            [pc])[0, true_class_index]
                        logging.warn("No classifier for '%s';"
                                     " using only global", connective_text)
                        pc_scores = [global_score, global_score, 0.0, 0.0]
                    else:
                        logging.warn("No classifier for '%s'; scoring as 0.0",
                                     connective_text)
                        pc_scores = [0.0] * 4
                scores.append(pc_scores)
            return scores
        else:
            # Predictions will be 1 or 0, which we can treat as just extreme
            # scores. (Don't worry about including all the scores, since there
            # are none of interest.)
            return [self.classifiers[stringify_connective(pc)].predict([pc])[0]
                    for pc in possible_causations]

class PatternBasedFilterDecoder(StructuredDecoder):
    def __init__(self, labels_for_eval, gold_labels_for_eval,
                 save_scored=False):
        super(PatternBasedFilterDecoder, self).__init__(save_scored)
        self._labels_for_eval = labels_for_eval
        self._gold_labels_for_eval = gold_labels_for_eval
        if save_scored:
            self.saved = []

    def decode(self, sentence, classifier_parts, scores):
        if not classifier_parts:
            return []

        if self.save_scored:
            self.saved.extend([
                (bool(p.possible_causation.true_causation_instance), probs)
                for p, probs in zip(classifier_parts, scores)])

        cutoff = FLAGS.filter_prob_cutoff
        tokens_to_parts = defaultdict(int)
        labels = [score[0] > cutoff for score in scores]
        self._labels_for_eval.extend(labels)
        self._gold_labels_for_eval.extend(
            [bool(p.possible_causation.true_causation_instance)
             for p in classifier_parts])

        # Deduplicate the results.
        positive_parts = [part for part, label in zip(classifier_parts, labels)
                          if label]
        for part in positive_parts:
            # Count every instance each connective word is part of.
            for connective_token in part.connective:
                tokens_to_parts[connective_token] += 1

        def should_keep_part(part):
            for token in part.connective:
                if tokens_to_parts[token] > 1:
                    # Assume that if there are other matches for a word, and
                    # this match relies on Steiner nodes, it's probably wrong.
                    # TODO: should we worry about cases where all connectives
                    # on this word were found using Steiner patterns?
                    if any('steiner_0' in pattern
                           for pattern in part.connective_patterns):
                        return False
                    # TODO: add check for duplicates in other cases?
            return True

        return [CausationInstance(sentence, connective=part.connective,
                                  cause=part.cause, effect=part.effect)
                for part in positive_parts
                if should_keep_part(part)]


class CausationPatternFilterStage(Stage):
    class FilterMetrics(ClassificationMetrics):
        def __init__(self, raw_classifier_metrics, *args, **kwargs):
            self.raw_classifier_metrics = raw_classifier_metrics
            super(self.FilterMetrics, self).__init__(*args, **kwargs)

        def __add__(self, other):
            metrics = object.__new__(type(self))
            metrics.__dict__ = super(type(self), self).__add__(
                other).__dict__
            try:
                metrics.raw_classifier_metrics = (
                    self.raw_classifier_metrics + other.raw_classifier_metrics)
            except AttributeError:
                metrics.raw_classifier_metrics = self.raw_classifier_metrics
            return metrics

        def __eq__(self, other):
            return (
                isinstance(other, self.FilterMetrics)
                and super(self.FilterMetrics, self).__eq__(other)
                and self.raw_classifier_metrics == other.raw_classifier_metrics)

        def __repr__(self):
            metrics_str = ClassificationMetrics.__repr__(self)
            raw_metrics_lines = str(self.raw_classifier_metrics).split('\n')
            to_join = [metrics_str, "Raw classifier metrics:"]
            for line in raw_metrics_lines:
                to_join.append('    ' + line)
            return '\n'.join(to_join)

        @staticmethod
        def average(metrics_list, ignore_nans=True):
            avg = object.__new__(CausationPatternFilterStage.FilterMetrics)
            avg.__dict__ = ClassificationMetrics.average(metrics_list,
                                                         ignore_nans).__dict__
            avg.raw_classifier_metrics = ClassificationMetrics.average(
                [m.raw_classifier_metrics for m in metrics_list], ignore_nans)
            return avg

    class ClassifierIAAEvaluator(IAAEvaluator):
        def __init__(self, decoder, *args, **kwargs):
            super(CausationPatternFilterStage.ClassifierIAAEvaluator, self
                  ).__init__(*args, **kwargs)
            self._decoder = decoder # for grabbing labels
            # Only convert without-partial metrics: if with-partial metrics are
            # present, the raw classification scores won't be any different.
            self._without_partial_metrics.connective_metrics = (
                self._convert_classification_metrics(
                    self._without_partial_metrics.connective_metrics))

        def evaluate(self, document, original_document, sentences,
                     original_sentences):
            super(CausationPatternFilterStage.ClassifierIAAEvaluator,
                  self).evaluate(document, original_document, sentences,
                                 original_sentences)
            if (FLAGS.filter_record_raw_accuracy
                and self._decoder._labels_for_eval):
                diff = diff_binary_vectors(self._decoder._labels_for_eval,
                                           self._decoder._gold_labels_for_eval,
                                           count_tns=False)
                metrics = self._without_partial_metrics
                metrics.connective_metrics.raw_classifier_metrics += diff

        def _convert_classification_metrics(self, classification_metrics):
            new_metrics = object.__new__(
                CausationPatternFilterStage.FilterMetrics)
            new_metrics.__dict__ = classification_metrics.__dict__
            new_metrics.raw_classifier_metrics = ClassificationMetrics(
                finalize=False)
            return new_metrics


        # Aggregation automatically handled by average() above.

    def __init__(self, classifier, name):
        self._labels_for_eval = []
        self._gold_labels_for_eval = []
        model = PatternBasedCausationFilter(classifier, self._labels_for_eval,
                                            self._gold_labels_for_eval)
        super(CausationPatternFilterStage, self).__init__(name=name,
                                                          model=model)

    def test(self, document, instances, writer=None):
        # Clear labels lists between documents.
        del self._labels_for_eval[:]
        del self._gold_labels_for_eval[:]
        super(CausationPatternFilterStage, self).test(document, instances,
                                                      writer=writer)

    consumed_attributes = ['possible_causations']

    def _label_instance(self, document, sentence, predicted_causations):
        sentence.causation_instances = predicted_causations

    def _make_evaluator(self):
        return self.ClassifierIAAEvaluator(self.model.decoder, False, False,
                                           FLAGS.filter_print_test_instances,
                                           True, True)


def get_weights_for_lr_classifier(classifier_pipeline, sort_weights=True):
    classifier = classifier_pipeline.steps[1][1]
    featurizer = classifier_pipeline.steps[0][1].featurizer
    feature_name_dict = featurizer.feature_name_dictionary

    if FLAGS.filter_feature_select_k == -1:
        # All features in feature dictionary are selected.
        feature_indices = range(len(feature_name_dict))
        lr = classifier
    else:
        feature_indices = classifier.named_steps[
            'feature_selection'].get_support().nonzero()[0]
        lr = classifier.named_steps['classification'].classifier

    weights = [(feature_name_dict.ids_to_names[ftnum], lr.coef_[0][i])
               for i, ftnum in enumerate(feature_indices)
               if lr.coef_[0][i] != 0.0]
    if sort_weights:
        # Sort by weights' absolute values.
        weights.sort(key=lambda tup: abs(tup[1]))
        return OrderedDict(weights)
    else:
        return dict(weights)
