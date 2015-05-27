from gflags import FLAGS, DuplicateFlagError, DEFINE_list, DEFINE_bool, DEFINE_integer
import logging

from causality_pipelines import IAAEvaluator
from data import Token, CausationInstance
from iaa import make_annotation_comparator
from pipeline import Stage
from pipeline.models import ClassifierPart, ClassifierModel
from pipeline.feature_extractors import FeatureExtractor, KnownValuesFeatureExtractor, SetValuedFeatureExtractor
from util.diff import SequenceDiff

try:
    DEFINE_list('regex_cc_features',
                ['cause_pos', 'effect_pos', 'wordsbtw', 'args_dep_path',
                 'args_dep_len', 'connective', 'tenses', 'pattern'],
                'Features for regex-based candidate classifier')
    DEFINE_integer('regex_cc_max_wordsbtw', 10,
                   "Maximum number of words between phrases before just making"
                   " the value the max")
    DEFINE_bool('regex_cc_log_differences', False,
                'Whether to print differing IAA results during evaluation')
    DEFINE_bool('regex_cc_train_with_partials', False,
                'Whether to train the regex-based candidate classifier model'
                ' counting partial overlap as correct')
    DEFINE_integer('regex_cc_max_dep_path_len', 3,
                   "Maximum number of dependency path steps to allow before just"
                   " making the value 'LONG-RANGE'")
except DuplicateFlagError as e:
    logging.warn('Ignoring redefinition of flag %s' % e.flagname)


class RegexClassifierPart(ClassifierPart):
    def __init__(self, possible_causation, label):
        sentence = possible_causation.sentence
        super(RegexClassifierPart, self).__init__(
            sentence, label)
        self.possible_causation = possible_causation

        self.cause_head = (sentence.get_head(possible_causation.cause)
                           if possible_causation.cause else None)
        self.effect_head = (sentence.get_head(possible_causation.effect)
                           if possible_causation.effect else None)

class RegexClassifierModel(ClassifierModel):
    def __init__(self, classifier):
        super(RegexClassifierModel, self).__init__(
            RegexClassifierPart,
            RegexClassifierModel.FEATURE_EXTRACTORS,
            FLAGS.regex_cc_features, classifier)

    @staticmethod
    def words_btw_heads(part):
        if part.cause_head is None or part.effect_head is None:
            return -1 # Out-of-band value for a natural number feature
        words_btw = part.instance.count_words_between(
            part.cause_head, part.effect_head)
        return min(words_btw, FLAGS.regex_cc_max_wordsbtw)

    @staticmethod
    def extract_dep_path(part):
        if part.cause_head is None or part.effect_head is None:
            return None
        deps = part.instance.extract_dependency_path(
            part.cause_head, part.effect_head, False)
        if len(deps) > FLAGS.regex_cc_max_dep_path_len:
            return 'LONG-RANGE'
        else:
            return str(deps)

    @staticmethod
    def extract_dep_path_length(part):
        if part.cause_head is None or part.effect_head is None:
            return -1 # Out-of-band value for a natural number feature
        deps = part.instance.extract_dependency_path(
            part.cause_head, part.effect_head, False)
        return min(len(deps), FLAGS.regex_cc_max_dep_path_len)

    __cached_tenses = {}
    __cached_tenses_sentence = None
    @staticmethod
    def extract_tense(head):
        if head is None:
            return "None"

        if (head.parent_sentence is
            RegexClassifierModel.__cached_tenses_sentence):
            try:
                return RegexClassifierModel.__cached_tenses[head]
            except KeyError:
                pass
        else:
            RegexClassifierModel.__cached_tenses_sentence = (
                head.parent_sentence)
            RegexClassifierModel.__cached_tenses = {}

        tense = head.parent_sentence.get_auxiliaries_string(head)
        RegexClassifierModel.__cached_tenses[head] = tense

        return tense

    # We can't initialize this properly yet because we don't have access to the
    # class' static methods to define the list.
    FEATURE_EXTRACTORS = []

RegexClassifierModel.FEATURE_EXTRACTORS = [
    KnownValuesFeatureExtractor(
        'cause_pos', lambda part: (part.cause_head.pos
                                   if part.cause_head else None),
        Token.ALL_POS_TAGS),
    KnownValuesFeatureExtractor(
        'effect_pos', lambda part: (part.effect_head.pos
                                    if part.effect_head else None),
        Token.ALL_POS_TAGS),
    # Generalized POS tags don't seem to be that useful.
    KnownValuesFeatureExtractor(
        'cause_pos_gen', lambda part: (part.cause_head.get_gen_pos()
                                       if part.cause_head else None),
        Token.ALL_POS_TAGS),
    KnownValuesFeatureExtractor(
        'effect_pos_gen', lambda part: (part.effect_head.get_gen_pos()
                                        if part.effect_head else None),
        Token.ALL_POS_TAGS),
    FeatureExtractor('wordsbtw', RegexClassifierModel.words_btw_heads,
                     FeatureExtractor.FeatureTypes.Numerical),
    FeatureExtractor('args_dep_path',
                     RegexClassifierModel.extract_dep_path),
    FeatureExtractor('args_dep_len',
                     RegexClassifierModel.extract_dep_path_length,
                     FeatureExtractor.FeatureTypes.Numerical),
    FeatureExtractor('connective',
                     lambda part: ' '.join(
                        [t.lemma for t in part.possible_causation.connective])),
    SetValuedFeatureExtractor(
        'patterns', lambda observation: observation.part.matching_patterns),
    FeatureExtractor('tenses',
                     lambda part: '/'.join(
                        [RegexClassifierModel.extract_tense(head)
                         for head in part.cause_head, part.effect_head]))]


class RegexClassifierStage(Stage):
    # TODO: add raw classifier evaluator (just going on what input was)
    def __init__(self, classifier, name):
        super(RegexClassifierStage, self).__init__(
            name, RegexClassifierModel(classifier))
        comparator = make_annotation_comparator(
            FLAGS.regex_cc_train_with_partials)
        # Comparator for matching CausationInstances against PossibleCausations
        self.connective_comparator = lambda inst1, inst2: comparator(
                                        inst1.connective, inst2.connective)

    def _make_evaluator(self):
        # TODO: provide both pairwise and non-pairwise stats.
        return IAAEvaluator(False, False, FLAGS.regex_cc_log_differences,
                            True, True)

    consumed_attributes = ['possible_causations']

    def _extract_parts(self, sentence, is_train):
        # In training, we need to match the causation instances the pipeline has
        # thus far detected against the original causation instances (provided
        # by previous pipeline stages). We do this the same way that the IAA
        # code does it internally: by running a diff on the connectives. Except
        # we cheat a bit, and compare PossibleCausations against real
        # CausationInstances.
        # TODO: try limiting to pairwise only for training.
        if is_train:
            parts = []
            # We want the diff to sort by connective position in the sentence.
            sort_by_key = lambda inst: inst.connective[0].start_offset
            connectives_diff = SequenceDiff(
                sentence.possible_causations, sentence.causation_instances,
                self.connective_comparator, sort_by_key)
            for correct_pc, _ in connectives_diff.get_matching_pairs():
                parts.append(RegexClassifierPart(correct_pc, True))
            for incorrect_pc in connectives_diff.get_a_only_elements():
                parts.append(RegexClassifierPart(incorrect_pc, False))
        else:
            # If we're not in training, the initial label doesn't really matter.
            parts = [RegexClassifierPart(pc, False)
                     for pc in sentence.possible_causations]
        return parts

    def _decode_labeled_parts(self, sentence, labeled_parts):
        sentence.causation_instances = [
            CausationInstance(
                sentence, None, None, part.possible_causation.connective,
                part.possible_causation.cause, part.possible_causation.effect)
            for part in labeled_parts if part.label]
