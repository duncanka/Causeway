from gflags import DEFINE_list, DEFINE_integer, DEFINE_bool, FLAGS, DuplicateFlagError

import logging

from causality_pipelines import IAAEvaluator
from data import Token
from pipeline import Stage
from pipeline.models import ClassifierModel, ClassifierPart
from pipeline.feature_extractors import KnownValuesFeatureExtractor, FeatureExtractor

try:
    DEFINE_list(
        'tregex_cc_features',
        ['cause_pos', 'effect_pos', 'wordsbtw', 'deppath', 'deplen', 'tenses',
         'connective', 'cn_daughter_deps', 'cn_incoming_dep', 'verb_children_deps',
         'cn_parent_pos', 'cn_words'],
        'Features to use for TRegex-based classifier model')
    DEFINE_integer('tregex_cc_max_wordsbtw', 10,
                   "TRegex-based classifier: maximum number of words between"
                   " phrases before just making the value the max")
    DEFINE_integer('tregex_cc_max_dep_path_len', 3,
                   "TRegex-based classifier: Maximum number of dependency path"
                   " steps to allow before just making the value 'LONG-RANGE'")
    DEFINE_bool('tregex_cc_print_test_instances', False,
                'TRegex-based: Whether to print differing IAA results'
                ' during evaluation')
except DuplicateFlagError as e:
    logging.warn('Ignoring redefinition of flag %s' % e.flagname)


class TRegexClassifierPart(ClassifierPart):
    def __init__(self, possible_causation):
        self.sentence = possible_causation.sentence
        label = possible_causation.true_causation_instance is not None
        super(TRegexClassifierPart, self).__init__(self.sentence, label)
        self.cause_head = self.sentence.get_head(possible_causation.cause)
        self.effect_head = self.sentence.get_head(possible_causation.effect)
        self.connective = possible_causation.connective
        self.connective_head = self.sentence.get_head(self.connective)
        # TODO: Update this once we have multiple pattern matches sorted out.
        self.connective_pattern = possible_causation.matching_patterns[0]


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
        edge_label, parent = part.sentence.get_most_direct_parent(
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

    # We can't initialize this properly yet because we don't have access to the
    # class' static methods to define the list.
    FEATURE_EXTRACTORS = []

TRegexClassifierModel.FEATURE_EXTRACTORS = [
    KnownValuesFeatureExtractor('cause_pos', lambda part: part.cause_head.pos,
                                Token.ALL_POS_TAGS),
    KnownValuesFeatureExtractor('effect_pos', lambda part: part.effect_head.pos,
                                Token.ALL_POS_TAGS),
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
    FeatureExtractor('cn_words', lambda part: ' '.join([t.original_text
                                                     for t in part.connective]))
]


class TRegexClassifierStage(Stage):
    def __init__(self, classifier, name):
        super(TRegexClassifierStage, self).__init__(
            name=name, models=[TRegexClassifierModel(classifier)])

    CONSUMED_ATTRIBUTES = ['possible_causations']

    def _extract_parts(self, sentence, is_train):
        return [TRegexClassifierPart(p) for p in sentence.possible_causations]

    def _decode_labeled_parts(self, sentence, labeled_parts):
        sentence.causation_instances = []
        for part in [p for p in labeled_parts if p.label]:
            # The only part type is phrase pair, so we don't have to worry
            # about checking the part type.
            sentence.add_causation_instance(
                connective=part.connective, cause=[part.cause_head],
                effect=[part.effect_head])

    def _make_evaluator(self):
        # TODO: provide both pairwise and non-pairwise stats
        return IAAEvaluator(False, False, FLAGS.tregex_cc_print_test_instances,
                            False, True, True)
