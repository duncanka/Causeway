from gflags import DEFINE_list, DEFINE_integer, FLAGS
import itertools
from sklearn import tree

from data import *
from pipeline import ClassifierStage
from pipeline.models import *

DEFINE_list('sc_features', ['pos1', 'pos2', 'wordsbtw'],
            'Features to use for simple causality model')
DEFINE_integer('sc_max_words_btw_phrases', 10,
    "Maximum number of words between phrases before just making the value '>'");

class PhrasePairPart(ClassifierPart):
    def __init__(self, sentence, head_token_1, head_token_2, label):
        super(PhrasePairPart, self).__init__(sentence, label)
        self.head_token_1 = head_token_1
        self.head_token_2 = head_token_2


class PhrasePairCausalityModel(ClassifierModel):
    FEATURE_EXTRACTOR_MAP = {
        'pos1': (True, lambda part: part.head_token_1.pos),
        'pos2': (True, lambda part: part.head_token_2.pos),
        'wordsbtw': (False,
                     lambda part: min(
                         # part.instance is the sentence.
                         part.instance.count_words_between(
                             part.head_token_1, part.head_token_2),
                         FLAGS.sc_max_words_btw_phrases))
    }

    def __init__(self):
        super(PhrasePairCausalityModel, self).__init__(
            PhrasePairPart, PhrasePairCausalityModel.FEATURE_EXTRACTOR_MAP,
            FLAGS.sc_features, tree.DecisionTreeClassifier())


class SimpleCausalityStage(ClassifierStage):
    def __init__(self):
        super(SimpleCausalityStage, self).__init__(
            'Simple causality', [PhrasePairCausalityModel()])
        self.expected_causations = []

    def _extract_parts(self, sentence):
        head_token_pairs = set(causation.get_cause_and_effect_heads()
                         for causation in sentence.causation_instances)
        clause_tokens = [token for token in sentence.tokens
                         if sentence.is_clause(token)]
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


    def _prepare_for_evaluation(self, sentences):
        self.expected_causations = [set(sentence.causation_instances)
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
                # None, if present as an argument, should be first.
                if (head_pair[0] is not None and (
                        head_pair[1] is None or
                        not instance.cause.starts_before(instance.effect))):
                    head_pair[0], head_pair[1] = head_pair[1], head_pair[0]

            return expected_heads == predicted_heads
            # End of causations_equivalent

        # Start of _evaluate
        for sentence, expected_causation_set in zip(sentences,
                                                    self.expected_causations):
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
                    self.fp += 1

            self.fn += len(expected_causation_set)

        self.expected_causations = []
