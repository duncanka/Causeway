from gflags import *
import itertools
import pexpect
import logging

from data import *
from pipeline import ClassifierStage
from pipeline.models import *
from util import Enum

try:
    DEFINE_list('sc_features', ['pos1', 'pos2', 'wordsbtw', 'deppath',
                                'deplen', 'tenses', 'tregexes'],
                'Features to use for simple causality model')
    DEFINE_integer('sc_max_words_btw_phrases', 10,
                   "Maximum number of words between phrases before just making"
                   " the value the max");
    DEFINE_integer('sc_max_dep_path_len', 3,
                   "Maximum number of dependency path steps to allow before"
                   " just making the value 'LONG-RANGE'");
    DEFINE_string('tregex_command',
                  '/home/jesse/Documents/Work/Research/'
                  'stanford-tregex-2014-10-26/tregex.sh',
                  'Command to run TRegex')
except DuplicateFlagError as e:
    logging.warn('Ignoring redefinition of flag %s' % e.flagname)

class PhrasePairPart(ClassifierPart):
    def __init__(self, sentence, head_token_1, head_token_2, label):
        super(PhrasePairPart, self).__init__(sentence, label)
        self.head_token_1 = head_token_1
        self.head_token_2 = head_token_2

class PhrasePairModel(ClassifierModel):
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
        if len(deps) > FLAGS.sc_max_dep_path_len:
            return 'LONG-RANGE'
        else:
            return str(deps)

    ConnectivePositions = Enum(['Before', 'Between', 'After'])

    @staticmethod
    def get_connective_position(connective, head1, head2):
        connective_start = connective.start_offset
        head1_start = head1.start_offset
        head2_start = head2.start_offset

        if connective_start < head1_start and connective_start < head2_start:
            return PhrasePairModel.ConnectivePositions.Before
        elif connective_start > head1_start and connective_start > head2_start:
            return PhrasePairModel.ConnectivePositions.After
        else: # one after and one before
            return PhrasePairModel.ConnectivePositions.Between

    @staticmethod
    def extract_connective_patterns(parts):
        connectives_seen = set()
        instances_seen = set()
        for part in parts:
            if part.instance in instances_seen:
                continue
            instances_seen.add(part.instance)

            for causation in part.instance.causation_instances:
                connective = causation.connective
                # Keep it simple for now: only single-word connectives.
                if len(connective) == 1:
                    connective = connective[0]
                    connective_text = connective.original_text
                    connective_position = (
                        PhrasePairModel.get_connective_position(
                            connective, part.head_token_1, part.head_token_2))
                    connectives_seen.add(
                        (connective_text, connective_position))

        return connectives_seen

    @staticmethod
    def make_connective_feature_extractors(connective_patterns):
        connective_features = []
        for connective_text, connective_position in connective_patterns:
            def extractor(part):
                for token in part.instance.tokens:
                    if (token.original_text == connective_text and
                        (PhrasePairModel.get_connective_position(
                                token, part.head_token_1, part.head_token_2)
                         == connective_position)):
                        return True
                return False
            feature_name = '%s_%s' % (
                connective_text,
                PhrasePairModel.ConnectivePositions[
                    connective_position])
            connective_features.append((feature_name, extractor))

        return connective_features

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

        tense = PhrasePairModel.__extract_tense_helper(head)
        PhrasePairModel.__cached_tenses[head] = tense
        return tense

    @staticmethod
    def __extract_tense_helper(head):
        # If it's not a copular construction and it's a noun phrase, the whole
        # argument is a noun phrase, so the notion of tense doesn't apply.
        copulas = head.parent_sentence.get_children(head, 'cop')
        if not copulas and head.pos in Token.NOUN_TAGS:
            return '<NOM>'

        auxiliaries = head.parent_sentence.get_children(head, 'aux')
        passive_auxes = head.parent_sentence.get_children(head, 'auxpass')
        auxes_plus_head = auxiliaries + passive_auxes + copulas + [head]
        auxes_plus_head.sort(key=lambda token: token.start_offset)

        CONTRACTION_DICT = {
            "'s": 'is',
             "'m": 'am',
             "'d": 'would',
             "'re": 'are',
             'wo': 'will',  # from "won't"
             'ca': 'can'  # from "can't"
        }
        aux_token_strings = []
        for token in auxes_plus_head:
            if token is head:
                aux_token_strings.append(token.pos)
            else:
                if token in copulas:
                    aux_token_strings.append('COP.' + token.pos)
                else:
                    aux_token_strings.append(
                         CONTRACTION_DICT.get(token.original_text,
                                              token.original_text))

        return '_'.join(aux_token_strings)

    @staticmethod
    def extract_tregex_patterns(parts):
        # TODO: Extend this code to multiple-word connectives and args.
        # TODO: Figure out tree transformations to get rid of dumb things like
        #       conjunctions that introduce spurious differences btw patterns?
        patterns = set()
        sentences_seen = set()

        def get_connective_arg_pattern(connective, arg, arg_name):
            parent_sentence = connective.parent_sentence
            dep_path = parent_sentence.extract_dependency_path(connective, arg)
            pattern = '=connective'
            last_node = connective

            for source, target, dep_name in dep_path:
                forward_dependency = source is last_node
                if forward_dependency:
                    next_node = target
                else:
                    next_node = source

                if next_node is arg:
                    node_name = '=' + arg_name
                else:
                    node_name = ''

                if parent_sentence.is_clause_head(next_node):
                    node_pos_pattern = '[<2 /^VB.*/ | < (__ <1 cop)]'
                else:
                    node_pos_pattern = ('<2 /^%s.*/' % next_node.get_gen_pos())

                if forward_dependency:
                    pattern = '%s < (__%s %s <1 %s' % (
                        pattern, node_name, node_pos_pattern, dep_name)
                else:
                    pattern = '%s <1 %s > (__%s %s' % (
                        pattern, dep_name, node_name, node_pos_pattern)

                last_node = next_node

            pattern += ')' * len(dep_path)
            return pattern

        for part in parts:
            parent_sentence = part.head_token_1.parent_sentence
            if parent_sentence in sentences_seen:
                continue
            sentences_seen.add(parent_sentence)

            for instance in parent_sentence.causation_instances:
                if (len(instance.connective) == 1 and instance.cause is not None
                    and instance.effect is not None):
                    connective = instance.connective[0]
                    cause_head = parent_sentence.get_head(instance.cause)
                    effect_head = parent_sentence.get_head(instance.effect)

                    cause_pattern = get_connective_arg_pattern(
                        connective, cause_head, 'cause')
                    effect_pattern = (
                        get_connective_arg_pattern(connective,
                                                   effect_head, 'effect'))

                    connective_pattern = (
                        '/^%s_[0-9]+$/=connective <2 /^%s.*/' % (
                            connective.lemma, connective.get_gen_pos()))

                    pattern = '%s : %s : %s' % (
                        connective_pattern, cause_pattern, effect_pattern)

                    #if pattern not in patterns:
                    #    print 'Adding pattern:\n\t%s\n\tSentence: %s\n' % (
                    #        pattern, parent_sentence.original_text)
                    patterns.add(pattern)

        return patterns

    __tregex_processes = {}
    __cached_tregex_results = {}
    __cached_ptb_str = None
    __cached_tregex_sentence = None
    @staticmethod
    def make_tregex_extractors(tregex_patterns):
        for process in PhrasePairModel.__tregex_processes.values():
            process.kill(9)
        PhrasePairModel.__tregex_processes.clear()

        '''
        Things to cache:
          1. Tree strings
          2. Results of running particular TRegexes on particular trees
        '''

        features = []
        for pattern in tregex_patterns:
            # To get rid of stderr junk, we need to redirect, which requires
            # spawning a shell.
            tregex_process = pexpect.spawn(
                'sh',
                ['-c', '%s -u -s -o -l -N -h cause -h effect -filter "" "%s"'
                 ' 2> /dev/null' % (FLAGS.tregex_command, pattern)])
            tregex_process.delaybeforesend = 0
            tregex_process.setecho(False)
            PhrasePairModel.__tregex_processes[pattern] = tregex_process

            def extractor(part):
                # Get the tree to pass to TRegex
                parent_sentence = part.head_token_1.parent_sentence
                if parent_sentence is not (
                    PhrasePairModel.__cached_tregex_sentence):
                    PhrasePairModel.__cached_tregex_sentence = parent_sentence
                    PhrasePairModel.__cached_tregex_results = {}
                    PhrasePairModel.__cached_ptb_str = (
                        parent_sentence.to_ptb_tree_string())

                try:
                    index_pairs = PhrasePairModel.__cached_tregex_results[pattern]
                except KeyError:
                    ptb_string = PhrasePairModel.__cached_ptb_str

                    # Interact with TRegex
                    tregex_process.sendline(ptb_string)
                    tregex_process.expect("\r\n\r\n") # look for the double newline
                    lines = tregex_process.before.split()[1:] # skip tree num line

                    # Parse TRegex output
                    line_pairs = zip(lines[0::2], lines[1::2])
                    index_pairs = [sorted([int(line.split("_")[-1])
                                           for line in line_pair])
                                    for line_pair in line_pairs]
                    PhrasePairModel.__cached_tregex_results[pattern] = (
                        index_pairs)

                for index_pair in index_pairs:
                    if index_pair == [part.head_token_1.index,
                                      part.head_token_2.index]:
                        return True

                return False

            features.append((pattern, extractor))
        return features


    # We can't initialize this properly yet because we don't have access to the
    # class' static methods to define the mapping.
    FEATURE_EXTRACTOR_MAP = {}

    def __init__(self, classifier):
        super(PhrasePairModel, self).__init__(
            PhrasePairPart,
            # Avoid any potential harm that could come to our class variable.
            PhrasePairModel.FEATURE_EXTRACTOR_MAP.copy(),
            FLAGS.sc_features, classifier)

Categorical = PhrasePairModel.FeatureTypes.Categorical
Numerical = PhrasePairModel.FeatureTypes.Numerical
PhrasePairModel.FEATURE_EXTRACTOR_MAP = {
    'pos1': (Categorical, lambda part: part.head_token_1.pos),
    'pos2': (Categorical, lambda part: part.head_token_2.pos),
    # Generalized POS tags don't seem to be that useful.
    'pos1gen': (Categorical, lambda part: part.head_token_1.get_gen_pos()),
    'pos2gen': (Categorical, lambda part: part.head_token_2.get_gen_pos()),
    'wordsbtw': (Numerical, PhrasePairModel.words_btw_heads),
    'deppath': (Categorical, PhrasePairModel.extract_dep_path),
    'deplen': (Numerical,
               lambda part: len(part.instance.extract_dependency_path(
                   part.head_token_1, part.head_token_2))),
    'connectives': (Categorical, TrainableFeatureExtractor(
            PhrasePairModel.extract_connective_patterns,
            PhrasePairModel.make_connective_feature_extractors)),
    'tenses': (Categorical, lambda part: '/'.join(
        [PhrasePairModel.extract_tense(head)
         for head in part.head_token_1, part.head_token_2])),
    # The tregexes feature is numerical in the sense that for each pattern it
    # extracts, the subfeature for that pattern is binary (present or not).
    'tregexes' : (Numerical, TrainableFeatureExtractor(
            PhrasePairModel.extract_tregex_patterns,
            PhrasePairModel.make_tregex_extractors))
}


class SimpleCausalityStage(ClassifierStage):
    @staticmethod
    def cause_starts_first(cause, effect):
        # None, if present as an argument, should be second.
        return effect is None or (
            cause is not None and
            effect.start_offset > cause.start_offset)

    def __init__(self, classifier):
        super(SimpleCausalityStage, self).__init__(
            'Simple causality', [PhrasePairModel(classifier)])
        self._expected_causations = []

    def _extract_parts(self, sentence):
        head_token_pairs = set(
            causation.get_cause_and_effect_heads(
                SimpleCausalityStage.cause_starts_first)
            for causation in sentence.causation_instances)
        clause_tokens = [token for token in sentence.tokens
                         # Only consider tokens that are in the parse tree.
                         if (sentence.get_depth(token) < len(sentence.tokens)
                             # Only consider clause or noun phrase heads.
                             and (sentence.is_clause_head(token)
                                  or token.pos in Token.NOUN_TAGS))]
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

            causation.cause = sentence.find_tokens_for_annotation(cause)
            causation.effect = sentence.find_tokens_for_annotation(effect)
            sentence.causation_instances.append(causation)

    def _begin_evaluation(self):
        super(SimpleCausalityStage, self)._begin_evaluation()
        self.tn = None

    def _prepare_for_evaluation(self, sentences):
        self._expected_causations = [set(sentence.causation_instances)
                                    for sentence in sentences]

    @staticmethod
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
        for head_pair in head_pairs:
            if not SimpleCausalityStage.cause_starts_first(*head_pair):
                head_pair[0], head_pair[1] = head_pair[1], head_pair[0]

        return expected_heads == predicted_heads

    def _evaluate(self, sentences):
        for sentence, expected_causation_set in zip(sentences,
                                                    self._expected_causations):
            for causation_instance in sentence.causation_instances:
                matching_expected_causation = None
                for expected_causation in expected_causation_set:
                    if self.causations_equivalent(expected_causation,
                                             causation_instance):
                        matching_expected_causation = expected_causation
                        self.tp += 1
                        break
                if matching_expected_causation:
                    expected_causation_set.remove(matching_expected_causation)
                else:
                    #print 'No match found for %s (cause: %s, effect: %s)' % (
                    #    causation_instance.source_sentence.original_text,
                    #    causation_instance.cause.original_text,
                    #    causation_instance.effect.original_text)
                    self.fp += 1

            self.fn += len(expected_causation_set)

        self._expected_causations = []
