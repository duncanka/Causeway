from gflags import DEFINE_list, DEFINE_integer, DEFINE_bool, FLAGS, DuplicateFlagError

import logging

from causality_pipelines import IAAEvaluator
from data import Token
from pipeline import Stage
from pipeline.models import ClassifierModel, ClassifierPart
from pipeline.feature_extractors import KnownValuesFeatureExtractor, FeatureExtractor

try:
    DEFINE_list(
        'pw_candidate_features', ['cause_pos', 'effect_pos', 'wordsbtw',
                                  'deppath', 'deplen', 'tenses', 'connective'],
        'Features to use for simple causality model')
    DEFINE_integer('pw_candidate_max_wordsbtw', 10,
                   "Pairwise classifier: maximum number of words between"
                   " phrases before just making the value the max")
    DEFINE_integer('pw_candidate_max_dep_path_len', 3,
                   "Pairwise classifier: Maximum number of dependency path steps"
                   " to allow before just making the value 'LONG-RANGE'")
    DEFINE_bool('pw_candidate_print_instances', False,
                'Pairwise classifier: Whether to print true positive, false'
                ' positive, and false negative instances after testing')
except DuplicateFlagError as e:
    logging.warn('Ignoring redefinition of flag %s' % e.flagname)


class PhrasePairPart(ClassifierPart):
    def __init__(self, possible_causation):
        sentence = possible_causation.sentence
        label = possible_causation.true_causation_instance is not None
        super(PhrasePairPart, self).__init__(sentence, label)
        self.cause_head = sentence.get_head(possible_causation.cause)
        self.effect_head = sentence.get_head(possible_causation.effect)
        self.connective = possible_causation.connective
        # TODO: Update this once we have multiple pattern matches sorted out.
        self.connective_pattern = possible_causation.matching_patterns[0]


class PhrasePairModel(ClassifierModel):
    def __init__(self, classifier):
        super(PhrasePairModel, self).__init__(
            PhrasePairPart,
            PhrasePairModel.FEATURE_EXTRACTORS,
            FLAGS.pw_candidate_features, classifier)

    @staticmethod
    def words_btw_heads(part):
        words_btw = part.instance.count_words_between(
            part.cause_head, part.effect_head)
        return min(words_btw, FLAGS.pw_candidate_max_wordsbtw)

    @staticmethod
    def extract_dep_path(part):
        deps = part.instance.extract_dependency_path(
            part.cause_head, part.effect_head, False)
        if len(deps) > FLAGS.pw_candidate_max_dep_path_len:
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
        if head.parent_sentence is PhrasePairModel.__cached_tenses_sentence:
            try:
                return PhrasePairModel.__cached_tenses[head]
            except KeyError:
                pass
        else:
            PhrasePairModel.__cached_tenses_sentence = head.parent_sentence
            PhrasePairModel.__cached_tenses = {}

        tense = head.parent_sentence.get_auxiliaries_string(head)
        PhrasePairModel.__cached_tenses[head] = tense
        return tense


    # We can't initialize this properly yet because we don't have access to the
    # class' static methods to define the list.
    FEATURE_EXTRACTORS = []

PhrasePairModel.FEATURE_EXTRACTORS = [
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
    FeatureExtractor('wordsbtw', PhrasePairModel.words_btw_heads,
                     FeatureExtractor.FeatureTypes.Numerical),
    FeatureExtractor('deppath', PhrasePairModel.extract_dep_path),
    FeatureExtractor('deplen',
                     lambda part: len(part.instance.extract_dependency_path(
                        part.cause_head, part.effect_head)),
                     FeatureExtractor.FeatureTypes.Numerical),
    # TODO: Update this once we have multiple pattern matches sorted out.
    FeatureExtractor('connective', lambda part: part.connective_pattern),
    FeatureExtractor('tenses',
                     lambda part: '/'.join(
                        [PhrasePairModel.extract_tense(head)
                         for head in part.cause_head, part.effect_head]))
]


class PairwiseCandidateClassifierStage(Stage):
    def __init__(self, classifier, name):
        super(PairwiseCandidateClassifierStage, self).__init__(
            name=name, models=[PhrasePairModel(classifier)])

    CONSUMED_ATTRIBUTES = ['possible_causations']

    def _extract_parts(self, sentence, is_train):
        return [PhrasePairPart(p) for p in sentence.possible_causations]

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
        return IAAEvaluator(False, False, FLAGS.pw_candidate_print_instances,
                            False, True, True)
