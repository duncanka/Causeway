from gflags import DEFINE_list, DEFINE_integer, FLAGS, DuplicateFlagError
import itertools
import logging

from data import *
from pipeline import ClassifierStage
from pipeline.models import *
from util import Enum

try:
    DEFINE_list('sc_features',
                ['pos1', 'pos2', 'wordsbtw', 'deplen'],
                'Features to use for simple causality model')
    DEFINE_integer('sc_max_words_btw_phrases', 10,
                   "Maximum number of words between phrases before just making"
                   " the value the max");
except DuplicateFlagError as e:
    logging.warn('Ignoring redefinition of flag %s' % e.flagname)

class PhrasePairPart(ClassifierPart):
    def __init__(self, sentence, head_token_1, head_token_2, label):
        super(PhrasePairPart, self).__init__(sentence, label)
        self.head_token_1 = head_token_1
        self.head_token_2 = head_token_2


class PhrasePairCausalityModel(ClassifierModel):
    # First define longer feature extraction functions.
    @staticmethod
    def words_btw_heads(part):
        words_btw = part.instance.count_words_between(
            part.head_token_1, part.head_token_2)
        return min(words_btw, FLAGS.sc_max_words_btw_phrases)

    @staticmethod
    def extract_dep_path(part):
        source = part.head_token_1
        target = part.head_token_2
        # Arbitrary convention to ensure that the string comes out the same
        # no matter which direction the dependency path goes: earlier start
        # offset is source.
        if source.start_offset > target.start_offset:
            source, target = target, source
        deps = part.instance.extract_dependency_path(source, target)
        return str(deps)

    ConnectivePositions = Enum(['Before', 'Between', 'After'])

    @staticmethod
    def get_connective_position(connective, head1, head2):
        connective_start = connective.offsets[0][0]
        head1_start = head1.start_offset
        head2_start = head2.start_offset

        if connective_start < head1_start and connective_start < head2_start:
            return PhrasePairCausalityModel.ConnectivePositions.Before
        elif connective_start > head1_start and connective_start > head2_start:
            return PhrasePairCausalityModel.ConnectivePositions.After
        else: # one after and one before
            return PhrasePairCausalityModel.ConnectivePositions.Between

    @staticmethod
    def extract_connective_patterns(parts):
        connectives_seen = set()
        for part in parts:
            for causation in part.instance.causation_instances:
                # Keep it simple for now: only single-word connectives.
                connective = causation.connective
                if len(connective.offsets) == 1:
                    connective_text = connective.text
                    connective_position = (
                        PhrasePairCausalityModel.get_connective_position(
                            connective, part.head_token_1, part.head_token_2))
                    connectives_seen.add(
                        (connective_text, connective_position))

        return connectives_seen

    @staticmethod
    def make_connective_feature_extractors(connective_patterns):
        connective_feature_map = {}
        for connective_text, connective_position in connective_patterns:
            def extractor(part):
                for token in part.instance.tokens:
                    if (token.text == connective_text and
                        (PhrasePairCausalityModel.get_connective_position(
                            token, part.head_token_1, part.head_token_2)
                         == connective_position)):
                        return True
                return False
            feature_name = '%s_%s' % (
                connective_text,
                PhrasePairCausalityModel.ConnectivePositions[
                    connective_position])
            connective_feature_map[feature_name] = extractor

        return connective_feature_map

    # We can't initialize this properly yet because we don't have access to the
    # class' static methods to define the mapping.
    FEATURE_EXTRACTOR_MAP = {}

    @staticmethod
    def cause_starts_first(cause, effect):
        # None, if present as an argument, should be second.
        return effect is None or (
            cause is not None and
            effect.start_offset > cause.start_offset)

    def __init__(self, classifier):
        super(PhrasePairCausalityModel, self).__init__(
            PhrasePairPart,
            # Avoid any potential harm that could come to our class variable.
            PhrasePairCausalityModel.FEATURE_EXTRACTOR_MAP.copy(),
            FLAGS.sc_features, classifier)

PhrasePairCausalityModel.FEATURE_EXTRACTOR_MAP = {
    'pos1': (True, lambda part: part.head_token_1.pos),
    'pos2': (True, lambda part: part.head_token_2.pos),
    # Generalized POS tags don't seem to be that useful.
    'pos1gen': (True, lambda part: ParsedSentence.POS_GENERAL.get(
        part.head_token_1.pos, part.head_token_1.pos)),
    'pos2gen': (True, lambda part: ParsedSentence.POS_GENERAL.get(
        part.head_token_2.pos, part.head_token_2.pos)),
    'wordsbtw': (False, PhrasePairCausalityModel.words_btw_heads),
    'deppath': (True, PhrasePairCausalityModel.extract_dep_path),
    'deplen': (False,
               lambda part: len(part.instance.extract_dependency_path(
                   part.head_token_1, part.head_token_2)))
}


class SimpleCausalityStage(ClassifierStage):
    def __init__(self, classifier):
        super(SimpleCausalityStage, self).__init__(
            'Simple causality', [PhrasePairCausalityModel(classifier)])
        self._expected_causations = []

    def _extract_parts(self, sentence):
        head_token_pairs = set(
            causation.get_cause_and_effect_heads(
                PhrasePairCausalityModel.cause_starts_first)
            for causation in sentence.causation_instances)
        clause_tokens = [token for token in sentence.tokens
                         # Only consider tokens that are in the parse tree.
                         if (sentence.get_depth(token) < len(sentence.tokens)
                             # Only consider clause or noun phrase heads.
                             and (sentence.is_clause_head(token)
                                  or token.pos in ParsedSentence.NOUN_TAGS))]
        return [PhrasePairPart(sentence, t1, t2, ((t1, t2) in head_token_pairs))
                for t1, t2 in itertools.combinations(clause_tokens, 2)]

    def _decode_labeled_parts(self, sentence, labeled_parts):
        sentence.causation_instances = []
        for part in [p for p in labeled_parts if p.label]:
            causation = CausationInstance(sentence)

            # The only part type is phrase pair, so we don't have to worry
            # about checking the part type.
            # We know it's a pair of phrases related by causation, so one is
            # the cause and one is the effect, but we don't actually know
            # which is which. We arbitrarily choose to call the one with the
            # earlier head the cause. We leave the connective unset.
            cause, effect = [
                Annotation(0, (head_token.start_offset, head_token.end_offset),
                           head_token.original_text)
                for head_token in (part.head_token_1, part.head_token_2)]
            if effect.starts_before(cause): # swap if they're ordered wrong
                effect, cause = cause, effect

            causation.cause = cause
            causation.effect = effect
            sentence.causation_instances.append(causation)

    def _begin_evaluation(self):
        super(SimpleCausalityStage, self)._begin_evaluation()
        self.tn = None

    def _prepare_for_evaluation(self, sentences):
        self._expected_causations = [set(sentence.causation_instances)
                                    for sentence in sentences]

    def _evaluate(self, sentences):
        def causations_equivalent(expected_instance, predicted_instance):
            """
            What it means for an expected instance to match a predicted instance
            is that the heads of the arguments match.
            """
            # Convert to lists to allow switching later.
            expected_heads = list(
                expected_instance.get_cause_and_effect_heads())
            predicted_heads = list(
                predicted_instance.get_cause_and_effect_heads())

            # Above, we arbitrarily established predicted cause is simply the
            # earliest argument, so we check whether the heads match in order of
            # appearance in the sentence, not whether cause and effect labels
            # match. (We make sure both lists of heads have the same convention
            # enforced on them, the same order on both, so it doesn't matter
            # which convention we use for which is the cause and which is the
            # effect.)
            head_pairs = [expected_heads, predicted_heads]
            instances = [expected_instance, predicted_instance]
            # Iterate over both expected and predicted to reorder both.
            for head_pair, instance in zip(head_pairs, instances):
                if not PhrasePairCausalityModel.cause_starts_first(*head_pair):
                    head_pair[0], head_pair[1] = head_pair[1], head_pair[0]

            return expected_heads == predicted_heads
            # End of causations_equivalent

        # Start of _evaluate
        for sentence, expected_causation_set in zip(sentences,
                                                    self._expected_causations):
            for causation_instance in sentence.causation_instances:
                matching_expected_causation = None
                for expected_causation in expected_causation_set:
                    if causations_equivalent(expected_causation,
                                             causation_instance):
                        matching_expected_causation = expected_causation
                        self.tp += 1
                        break
                if matching_expected_causation:
                    expected_causation_set.remove(matching_expected_causation)
                else:
                    #print 'No match found for %s (cause: %s, effect: %s)' % (
                    #    causation_instance.source_sentence.original_text,
                    #    causation_instance.cause.text, causation_instance.effect.text)
                    self.fp += 1

            self.fn += len(expected_causation_set)

        self._expected_causations = []
