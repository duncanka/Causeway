from collections import defaultdict
from gflags import DEFINE_list, DEFINE_integer, DEFINE_bool, FLAGS, DuplicateFlagError
from itertools import chain, product
import logging
from nltk.corpus import wordnet
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
    VectorValuedFeatureExtractor, FeaturizationError)
from pipeline.models.structured import StructuredDecoder, StructuredModel
from skpipeline import (make_featurizing_estimator,
                        make_mostfreq_featurizing_estimator)
from util.diff import SequenceDiff
from util.scipy import AutoWeightedVotingClassifier


try:
    DEFINE_list(
        'filter_features',
        'cause_pos,effect_pos,wordsbtw,deppath,deplen,connective,cn_lemmas,'
        'tenses,cause_case_children,effect_case_children,domination,'
        'cause_pp_prep,effect_pp_prep,cause_neg,effect_neg,commas_btw,'
        'cause_prep_start,effect_prep_start'.split(','),
        'Features to use for pattern-based candidate classifier model')
    DEFINE_integer('filter_max_wordsbtw', 10,
                   "Maximum number of words between phrases before just making"
                   " the value the max")
    DEFINE_integer('filter_max_dep_path_len', 3,
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
    DEFINE_integer('filter_feature_select_k', 100,
                   "Specifies how many features to keep in feature selection"
                   " for per-connective causality filters. -1 means no feature"
                   " selection.")
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


class CausalPatternClassifierModel(object):
    def __init__(self, classifier, selected_features=None,
                 model_path=None, save_featurized=False):
        super(CausalPatternClassifierModel, self).__init__(
            classifier=classifier, selected_features=selected_features,
            model_path=model_path, save_featurized=save_featurized)

    @staticmethod
    def _get_gold_labels(classifier_parts):
        return [part.connective_correct for part in classifier_parts]

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
            CausalPatternClassifierModel.__cached_tenses_sentence):
            try:
                return CausalPatternClassifierModel.__cached_tenses[head]
            except KeyError:
                pass
        else:
            CausalPatternClassifierModel.__cached_tenses_sentence = (
                head.parent_sentence)
            CausalPatternClassifierModel.__cached_tenses = {}

        tense = head.parent_sentence.get_auxiliaries_string(head)
        CausalPatternClassifierModel.__cached_tenses[head] = tense
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
        _edge_label, parent = part.sentence.get_most_direct_parent(
            part.connective_head)
        if parent is None:
            return None
        return CausalPatternClassifierModel.get_pos_with_copulas(parent)

    _ALL_POS_PAIRS = ['/'.join(tags) for tags in product(
                        Token.ALL_POS_TAGS, Token.ALL_POS_TAGS)]
    @staticmethod
    def extract_pos_bigram(part, arg_head):
        if arg_head.index < 2:
            prev_pos = 'NONE'
            pos = CausalPatternClassifierModel.get_pos_with_copulas(arg_head)
        else:
            previous_token = part.sentence.tokens[arg_head.index - 1]
            # TODO: would this be helpful or harmful?
            # while previous_token.pos in Token.PUNCT_TAGS:
            #     previous_token = part.sentence.tokens[
            #         previous_token.index - 1]
            prev_pos, pos = [CausalPatternClassifierModel.get_pos_with_copulas(
                                token) for token in previous_token, arg_head]
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
        if not CausalPatternClassifierModel._embeddings:
            CausalPatternClassifierModel._embeddings = SennaEmbeddings()
        try:
            return CausalPatternClassifierModel._embeddings[
                arg_head.lowered_text]
        except KeyError: # Unknown word; return special vector
            return CausalPatternClassifierModel._embeddings['UNKNOWN']

    @staticmethod
    def extract_vector_dist(head1, head2):
        v1 = CausalPatternClassifierModel.extract_vector(head1)
        v2 = CausalPatternClassifierModel.extract_vector(head2)
        return np.linalg.norm(v1 - v2)

    @staticmethod
    def extract_vector_cos_dist(head1, head2):
        v1 = CausalPatternClassifierModel.extract_vector(head1)
        v2 = CausalPatternClassifierModel.extract_vector(head2)
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
    def is_negated(argument_head):
        sentence = argument_head.parent_sentence
        children = sentence.get_children(argument_head, 'neg')
        return bool(children)

    @staticmethod
    def is_comp(argument_head):
        sentence = argument_head.parent_sentence
        edge_label, _ = sentence.get_most_direct_parent(argument_head)
        if edge_label != 'ccomp':
            return False
        else:
            children = sentence.get_children(argument_head, 'mark')
            return bool(children)

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

    all_feature_extractors = []


Numerical = FeatureExtractor.FeatureTypes.Numerical

CausalPatternClassifierModel.connective_feature_extractors = [
    SetValuedFeatureExtractor(
        'connective', lambda part: part.connective_patterns),
    FeatureExtractor('cn_words',
                     lambda part: ' '.join([t.lowered_text
                                            for t in part.connective])),
    FeatureExtractor('cn_lemmas',
                     lambda part: ' '.join([t.lemma
                                            for t in part.connective])), ]

CausalPatternClassifierModel.general_feature_extractors = [
    KnownValuesFeatureExtractor(
        'cause_pos',
        lambda part: CausalPatternClassifierModel.get_pos_with_copulas(
                        part.cause_head),
        Token.ALL_POS_TAGS),
    KnownValuesFeatureExtractor(
        'effect_pos',
        lambda part: CausalPatternClassifierModel.get_pos_with_copulas(
                        part.effect_head),
        Token.ALL_POS_TAGS),
    KnownValuesFeatureExtractor(
        'cause_pos_bigram',
        lambda part: CausalPatternClassifierModel.extract_pos_bigram(
                             part, part.cause_head),
                         CausalPatternClassifierModel._ALL_POS_PAIRS),
    KnownValuesFeatureExtractor(
        'effect_pos_bigram',
        lambda part: CausalPatternClassifierModel.extract_pos_bigram(
                             part, part.effect_head),
                         CausalPatternClassifierModel._ALL_POS_PAIRS),
    # Generalized POS tags don't seem to be that useful.
    KnownValuesFeatureExtractor(
        'cause_pos_gen', lambda part: part.cause_head.get_gen_pos(),
        Token.ALL_POS_TAGS),
    KnownValuesFeatureExtractor(
        'effect_pos_gen', lambda part: part.effect_head.get_gen_pos(),
        Token.ALL_POS_TAGS),
    FeatureExtractor('wordsbtw', CausalPatternClassifierModel.words_btw_heads,
                     Numerical),
    FeatureExtractor('deppath', CausalPatternClassifierModel.extract_dep_path),
    FeatureExtractor('deplen',
                     lambda part: len(part.sentence.extract_dependency_path(
                        part.cause_head, part.effect_head)), Numerical),
    FeatureExtractor('tenses',
                     lambda part: '/'.join(
                        [CausalPatternClassifierModel.extract_tense(head)
                         for head in part.cause_head, part.effect_head])),
    FeatureExtractor('cn_daughter_deps',
                     CausalPatternClassifierModel.extract_daughter_deps),
    FeatureExtractor('cn_incoming_dep',
                     CausalPatternClassifierModel.extract_incoming_dep),
    FeatureExtractor('verb_children_deps',
                     CausalPatternClassifierModel.get_verb_children_deps),
    FeatureExtractor('cn_parent_pos',
                     CausalPatternClassifierModel.extract_parent_pos),
    SetValuedFeatureExtractor(
        'cause_hypernyms',
        lambda part: CausalPatternClassifierModel.extract_wn_hypernyms(
            part.cause_head)),
    SetValuedFeatureExtractor(
        'effect_hypernyms',
        lambda part: CausalPatternClassifierModel.extract_wn_hypernyms(
            part.effect_head)),
    FeatureExtractor(
        'cause_case_children',
        lambda part: CausalPatternClassifierModel.extract_case_children(
                        part.cause_head)),
    FeatureExtractor('effect_case_children',
        lambda part: CausalPatternClassifierModel.extract_case_children(
                        part.effect_head)),
    KnownValuesFeatureExtractor('domination',
        lambda part: part.sentence.get_domination_relation(
        part.cause_head, part.effect_head),
        range(len(StanfordParsedSentence.DOMINATION_DIRECTION))),
    VectorValuedFeatureExtractor(
        'cause_vector',
        lambda part: CausalPatternClassifierModel.extract_vector(
                        part.cause_head)),
    VectorValuedFeatureExtractor(
        'effect_vector',
        lambda part: CausalPatternClassifierModel.extract_vector(
                        part.cause_head)),
    FeatureExtractor(
        'vector_dist',
        lambda part: CausalPatternClassifierModel.extract_vector_dist(
                         part.cause_head, part.effect_head), Numerical),
    FeatureExtractor(
        'vector_cos_dist',
        lambda part: CausalPatternClassifierModel.extract_vector_cos_dist(
                        part.cause_head, part.effect_head), Numerical),
    KnownValuesFeatureExtractor(
        'ners', lambda part: '/'.join(
                    StanfordNERStage.NER_TYPES[arg_head.ner_tag]
                    for arg_head in [part.cause_head, part.effect_head]),
        StanfordNERStage.NER_TYPES),
    FeatureExtractor(
        'commas_btw',
        lambda part: CausalPatternClassifierModel.count_commas_between(
            part.cause, part.effect), Numerical),
    FeatureExtractor('cause_len', lambda part: len(part.cause), Numerical),
    FeatureExtractor('effect_len', lambda part: len(part.effect), Numerical),
    FeatureExtractor(
        'args_rel_pos',
        lambda part: CausalPatternClassifierModel.cause_pos_wrt_effect(
            part.cause, part.effect)),
    FeatureExtractor(
        'heads_rel_pos',
        lambda part: CausalPatternClassifierModel.cause_pos_wrt_effect(
            [part.cause_head], [part.effect_head])),
    SetValuedFeatureExtractor(
        'cause_pp_prep',
        lambda part: CausalPatternClassifierModel.get_pp_preps(
            part.cause_head)),
    SetValuedFeatureExtractor(
        'effect_pp_prep',
        lambda part: CausalPatternClassifierModel.get_pp_preps(
            part.effect_head)),
    FeatureExtractor(
        'cause_neg',
        lambda part: CausalPatternClassifierModel.is_negated(part.cause_head),
        Numerical),
    FeatureExtractor(
        'effect_neg',
        lambda part: CausalPatternClassifierModel.is_negated(part.effect_head),
        Numerical),
    FeatureExtractor(
        'cause_comp',
        lambda part: CausalPatternClassifierModel.is_comp(part.cause_head),
        Numerical),
    FeatureExtractor(
        'effect_comp',
        lambda part: CausalPatternClassifierModel.is_comp(part.effect_head),
        Numerical),
    FeatureExtractor(
        'cause_comp_start',
        lambda part: CausalPatternClassifierModel.starts_w_comp(part.cause),
        Numerical),
    FeatureExtractor(
        'effect_comp_start',
        lambda part: CausalPatternClassifierModel.starts_w_comp(part.effect),
        Numerical),
    FeatureExtractor(
        'cause_prep_start',
        lambda part: CausalPatternClassifierModel.initial_prep(part.cause)),
    FeatureExtractor(
        'effect_prep_start',
        lambda part: CausalPatternClassifierModel.initial_prep(part.effect)),
]

CausalPatternClassifierModel.all_feature_extractors = (
    CausalPatternClassifierModel.connective_feature_extractors
    + CausalPatternClassifierModel.general_feature_extractors)


class PatternBasedCausationFilter(StructuredModel):
    def __init__(self, classifier, save_featurized=False):
        super(PatternBasedCausationFilter, self).__init__(
            PatternBasedFilterDecoder())
        
        if FLAGS.filter_feature_select_k == -1:
            base_per_conn_classifier = classifier
            general_classifier = classifier
        else:
            k_best = SelectKBest(chi2, FLAGS.filter_feature_select_k)
            base_per_conn_classifier = SKLPipeline([
                ('feature_selection', k_best),
                ('classification', classifier)
            ])
            general_classifier = sklearn.clone(base_per_conn_classifier)

        all_extractors = CausalPatternClassifierModel.all_feature_extractors
        general_extractors = (
            CausalPatternClassifierModel.general_feature_extractors)
        all_extractor_names = set([e.name for e in all_extractors] + ['all'])
        for selected_name in FLAGS.filter_features:
            if selected_name not in all_extractor_names:
                raise FeaturizationError('Invalid feature name: %s'
                                         % selected_name)

        self.base_per_conn_classifier = make_featurizing_estimator(
            base_per_conn_classifier,
            'causality_pipelines.candidate_filter.CausalPatternClassifierModel'
            '.all_feature_extractors',
            self._get_selected_features_for_extractors(FLAGS.filter_features,
                                                       all_extractors),
            'per_conn_classifier')
        self.general_classifier = make_featurizing_estimator(
            general_classifier,
            'causality_pipelines.candidate_filter.CausalPatternClassifierModel'
            '.general_feature_extractors',
            self._get_selected_features_for_extractors(FLAGS.filter_features,
                                                       general_extractors),
            'global_causality_classifier')
        self.base_mostfreq_classifier = make_mostfreq_featurizing_estimator(
            'most_freq_classifier')

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

    @staticmethod
    def _get_selected_features_for_extractors(selected_features, extractors):
        extractor_names = [e.name for e in extractors] + ['all']
        return [name for name in selected_features if name in extractor_names]

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

    def _train_structured(self, sentences, parts_by_sentence):
        all_pcs = list(chain.from_iterable(parts_by_sentence))
        all_labels = CausalPatternClassifierModel._get_gold_labels(all_pcs)
        self.general_classifier.fit(all_pcs, all_labels)

        pcs_by_connective = defaultdict(list)
        for pc in all_pcs:
            connective = stringify_connective(pc)
            pcs_by_connective[connective].append(pc)

        for connective, pcs in pcs_by_connective.iteritems():
            pcs_with_both_args = [pc for pc in pcs if pc.cause and pc.effect]
            pcs = pcs_with_both_args # train only on instances with 2 args
            labels = CausalPatternClassifierModel._get_gold_labels(pcs)

            # Some classifiers don't deal well with all labels being the same.
            # If this is the case, it should just default to majority class
            # anyway, so just do that.
            if len(set(labels)) < 2:
                classifier = make_mostfreq_featurizing_estimator()
                classifier.fit(pcs, labels)
            else:
                per_conn = sklearn.clone(self.base_per_conn_classifier)
                mostfreq = sklearn.clone(self.base_mostfreq_classifier)
                for new_classifier in per_conn, mostfreq:
                    try:
                        new_classifier.fit(pcs, labels)
                    except ValueError:
                        classification_pipeline = new_classifier.steps[1][1]
                        feature_selector = classification_pipeline.steps[0][1]
                        feature_selector.k = 'all'
                        new_classifier.fit(pcs, labels)

                classifier = AutoWeightedVotingClassifier(
                    estimators=[('per_conn', per_conn), ('mostfreq', mostfreq),
                                ('global_causality', self.general_classifier)],
                    voting='soft')
                classifier.fit_weights(pcs, labels)

            self.classifiers[connective] = classifier

    def _score_parts(self, sentence, possible_causations):
        return [self.classifiers[stringify_connective(pc)].predict([pc])
                for pc in possible_causations]


class PatternBasedFilterDecoder(StructuredDecoder):
    def decode(self, sentence, classifier_parts, labels):
        # Deduplicate the results.

        tokens_to_parts = defaultdict(int)
        positive_parts = [part for part, label in zip(classifier_parts, labels)
                          if label]
        for part in positive_parts:
            # Count every instance each connective word is part of.
            for connective_token in part.connective:
                tokens_to_parts[connective_token] += 1

        causation_instances = []
        for part in positive_parts:
            keep_part = True
            for token in part.connective:
                if tokens_to_parts[token] > 1:
                    # Assume that if there are other matches for a word, and
                    # this match relies on Steiner nodes, it's probably wrong.
                    # TODO: should we worry about cases where all connectives
                    # on this word were found using Steiner patterns?
                    if any('steiner_0' in pattern
                           for pattern in part.connective_patterns):
                        keep_part = False
                        break
                    # TODO: add check for duplicates in other cases?
            if keep_part:
                causation_instances.append(CausationInstance(
                    sentence, connective=part.connective,
                    cause=part.cause, effect=part.effect))

        return causation_instances


class CausationPatternFilterStage(Stage):
    def __init__(self, classifier, name):
        super(CausationPatternFilterStage, self).__init__(
            name=name, model=PatternBasedCausationFilter(classifier))

    consumed_attributes = ['possible_causations']

    def _label_instance(self, document, sentence, predicted_causations):
        sentence.causation_instances = predicted_causations

    def _make_evaluator(self):
        return IAAEvaluator(False, False,
                            FLAGS.filter_print_test_instances, True, True)
