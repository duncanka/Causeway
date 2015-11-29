from __future__ import print_function
from collections import defaultdict
import colorama
colorama.init()
colorama.deinit()
from copy import copy
from StringIO import StringIO
from gflags import DEFINE_list, DEFINE_float, DEFINE_bool, DEFINE_string, DuplicateFlagError, FLAGS
import logging
import numpy as np
import operator
import os
import sys
from textwrap import wrap

from data import CausationInstance, StanfordParsedSentence, Token
from util import Enum, print_indented, truncated_string, get_terminal_size
from util.diff import SequenceDiff
from util.metrics import ClassificationMetrics, ConfusionMatrix, AccuracyMetrics, \
    safe_divisor

np.seterr(divide='ignore') # Ignore nans in division

try:
    DEFINE_list(
        'iaa_given_connective_ids', [], "Annotation IDs for connectives"
        " that were given as gold and should be treated separately for IAA.")
    DEFINE_float(
        'iaa_min_partial_overlap', 0.5, "Minimum fraction of the larger of two"
        " annotations that must be overlapping for the two annotations to be"
        " considered a partial match.")
    DEFINE_bool('iaa_log_confusion', False, "Log confusion matrices for IAA.")
    DEFINE_bool('iaa_log_stats', True, 'Log IAA statistics.')
    DEFINE_bool('iaa_log_differences', False,
                'Log differing annotations during IAA comparison.')
    DEFINE_bool('iaa_log_agreements', False,
                'Log agreeing annotations during IAA comparison.')
    DEFINE_string('iaa_cause_color', 'Blue',
                  'ANSI color to use for formatting cause words in IAA'
                  ' comparison output')
    DEFINE_string('iaa_effect_color', 'Red',
                  'ANSI color to use for formatting cause words in IAA'
                  ' comparison output')
    DEFINE_bool('iaa_force_color', False,
                "Force ANSI color in IAA comparisons even when we're not"
                " outputting to a TTY")
    DEFINE_bool('iaa_check_punct', False,
                'Whether IAA should compare punctuation tokens to determine'
                ' argument matches')
except DuplicateFlagError as e:
    logging.warn('Ignoring redefinition of flag %s' % e.flagname)


def make_annotation_comparator(allow_partial):
    min_partial_overlap = [1.0, FLAGS.iaa_min_partial_overlap][allow_partial]

    def match_annotations(token_list_1, token_list_2):
        offsets_1 = [(token.start_offset, token.end_offset)
                     for token in token_list_1]
        offsets_2 = [(token.start_offset, token.end_offset)
                     for token in token_list_2]
        if offsets_1 == offsets_2:
            return True

        # No partial matching allowed
        if min_partial_overlap == 1.0:
            return False

        a1_length = sum([end - start for start, end in offsets_1])
        a2_length = sum([end - start for start, end in offsets_2])
        # Larger length is converted to float below to avoid having to do that
        # repeatedly when computing fractions.
        if a1_length > a2_length:
            larger_offsets, smaller_offsets, larger_length = (
                offsets_1, offsets_2, float(a1_length))
        else:
            larger_offsets, smaller_offsets, larger_length = (
                offsets_2, offsets_1, float(a2_length))

        # This algorithm assumes no fragments ever overlap with each other.
        # Thus, each token of the smaller annotation can only ever overlap with
        # a single fragment from the larger annotation. This means we can safely
        # add a separate fraction to the total percent overlap each time we
        # detect any overlap at all.
        fraction_of_larger_overlapping = 0.0
        for larger_offset in larger_offsets:
            for smaller_offset in smaller_offsets:
                overlap_start = max(larger_offset[0], smaller_offset[0])
                overlap_end = min(larger_offset[1], smaller_offset[1])
                overlap_size = overlap_end - overlap_start
                if overlap_size > 0:
                    fraction_of_larger_overlapping += (
                        overlap_size / larger_length)

        return fraction_of_larger_overlapping > min_partial_overlap

    return match_annotations
compare_annotations_partial = make_annotation_comparator(True)
compare_annotations_exact = make_annotation_comparator(False)

def _get_printable_connective_word(word):
    if sys.stdout.isatty() or FLAGS.iaa_force_color:
        return colorama.Format.BOLD + word.upper() + colorama.Format.RESET_ALL
    else:
        return word.upper()

def _wrapped_sentence_highlighting_instance(instance):
    # TODO: use GFlags wrapping machinery?
    sentence = instance.sentence
    words = [(_get_printable_connective_word(t.original_text)
              if t in instance.connective else t.original_text)
             for t in sentence.tokens[1:]]
    lines = wrap(' '.join(words), 100, subsequent_indent='    ',
                 break_long_words=False)
    return '\n'.join(lines)

class CausalityMetrics(object):
    IDsConsidered = Enum(['GivenOnly', 'NonGivenOnly', 'Both'])
    _ARG_ATTR_NAMES = ['cause_span_metrics', 'effect_span_metrics',
                       'cause_head_metrics', 'effect_head_metrics',
                       'cause_jaccard', 'effect_jaccard']

    # TODO: Refactor order of parameters
    def __init__(
        self, gold, predicted, allow_partial, save_differences=False,
        ids_considered=None, compare_degrees=True, compare_types=True,
        compare_args=True, pairwise_only=False, save_agreements=False,
        causations_property_name='causation_instances'):

        assert len(gold) == len(predicted), (
            "Cannot compute IAA for different-sized datasets")

        if ids_considered is None:
            ids_considered = CausalityMetrics.IDsConsidered.Both
        self.allow_partial = allow_partial
        self._annotation_comparator = make_annotation_comparator(allow_partial)
        self.ids_considered = ids_considered
        self.save_differences = save_differences
        self.save_agreements = save_agreements
        self.pairwise_only = pairwise_only
        self.causations_property_name = causations_property_name

        self.gold_only_instances = []
        self.predicted_only_instances = []
        self.agreeing_instances = []
        self.argument_differences = []
        self.property_differences = []

        # Compute attributes that take a little more work.
        self.connective_metrics, matches = self._match_connectives(
            gold, predicted)
        if compare_degrees:
            self.degree_matrix = self._compute_agreement_matrix(
                matches, CausationInstance.Degrees, 'degree', gold)
        else:
            self.degree_matrix = None

        if compare_types:
            self.causation_type_matrix = self._compute_agreement_matrix(
                matches, CausationInstance.CausationTypes, 'type', gold)
        else:
            self.causation_type_matrix = None

        # TODO: add back metrics that account for the possibility of null args
        # (i.e., P/R/F1).
        if compare_args:
            (self.cause_span_metrics, self.effect_span_metrics,
             self.cause_head_metrics, self.effect_head_metrics) = (
                self._match_arguments(matches, gold))
    
            self.cause_jaccard = self._get_jaccard(matches, 'cause')
            self.effect_jaccard = self._get_jaccard(matches, 'effect')
        else:
            for attr_name in self._ARG_ATTR_NAMES:
                setattr(self, attr_name, None)

    def __add__(self, other):
        if (self.allow_partial != other.allow_partial or
            [self.degree_matrix, other.degree_matrix].count(None) == 1 or
            [self.causation_type_matrix,
             other.causation_type_matrix].count(None) == 1):
            raise ValueError("Can't add causality metrics with different"
                             " comparison criteria")

        sum_metrics = copy(self)
        # Add recorded instances/differences
        sum_metrics.gold_only_instances.extend(other.gold_only_instances)
        sum_metrics.predicted_only_instances.extend(
            other.predicted_only_instances)
        sum_metrics.property_differences.extend(other.property_differences)
        sum_metrics.agreeing_instances.extend(other.agreeing_instances)
        # Add together submetrics, if they exist
        sum_metrics.connective_metrics += other.connective_metrics
        for attr_name in (['degree_matrix', 'causation_type_matrix']
                          + self._ARG_ATTR_NAMES):
            self_attr = getattr(self, attr_name)
            other_attr = getattr(other, attr_name)
            if self_attr is not None and other_attr is not None:
                # Ignore all NaNs -- just pretend they're not even there.
                if isinstance(self_attr, float) and np.isnan(self_attr):
                    attr_value = other_attr
                elif isinstance(other_attr, float) and np.isnan(other_attr):
                    attr_value = self_attr
                else:
                    if attr_name.endswith('jaccard'):
                        attr_value = (self_attr + other_attr) / 2.0
                    else:
                        attr_value = self_attr + other_attr
            else:
                attr_value = None
            setattr(sum_metrics, attr_name, attr_value)

        return sum_metrics

    def __get_causations(self, sentence, is_gold):
        causations = []
        property_name = [self.causations_property_name,
                         'causation_instances'][is_gold]
        for instance in getattr(sentence, property_name):
            if (# First set of conditions: matches givenness specified
                (self.ids_considered == self.IDsConsidered.Both or
                 (instance.id in FLAGS.iaa_given_connective_ids and
                  self.ids_considered == self.IDsConsidered.GivenOnly) or
                 (instance.id not in FLAGS.iaa_given_connective_ids and
                  self.ids_considered == self.IDsConsidered.NonGivenOnly))
                # Second set of conditions: is pairwise if necessary
                and (not is_gold or not self.pairwise_only or
                     (instance.cause != None and instance.effect != None))):
                causations.append(instance)
        return causations

    @staticmethod
    def get_connective_matches(gold_causations, predicted_causations,
                               allow_partial):
        def compare_connectives_exact(instance_1, instance_2):
            return compare_annotations_exact(
                instance_1.connective, instance_2.connective)
        if allow_partial:
            def compare_connectives(instance_1, instance_2):
                return compare_annotations_partial(
                    instance_1.connective, instance_2.connective)
        else:
            compare_connectives = compare_connectives_exact
        # Sort instances in case they're somehow out of order, or there are
        # multiple annotations with the same connective that may be unordered.
        # TODO: is this sufficient? Do we need to worry about, e.g., ordering by
        # head, or what happens if the cause is None?
        sort_key = lambda inst: (
            inst.connective[0].start_offset,
            inst.cause[0].start_offset if inst.cause else 0,
            inst.effect[0].start_offset if inst.effect else 0)

        matching_instances = []
        gold_only_instances = []
        predicted_only_instances = []

        # If we're allowing partial matches, we don't want any partial
        # matches to override full matches. So we first do an exact match,
        # and remove the ones that matched from the partial matching.
        if allow_partial:
            diff = SequenceDiff(gold_causations, predicted_causations,
                                compare_connectives_exact, sort_key)
            matching_pairs = diff.get_matching_pairs()
            matching_instances.extend(matching_pairs)
            # Instances that were gold-only or predicted-only may still generate
            # partial matches.
            gold_causations = diff.get_a_only_elements()
            predicted_causations = diff.get_b_only_elements()

        diff = SequenceDiff(gold_causations, predicted_causations,
                            compare_connectives, sort_key)
        matching_instances.extend(diff.get_matching_pairs())
        gold_only_instances.extend(diff.get_a_only_elements())
        predicted_only_instances.extend(diff.get_b_only_elements())

        return matching_instances, gold_only_instances, predicted_only_instances

    def _match_connectives(self, gold, predicted):
        matching_instances = []
        gold_only_instances = []
        predicted_only_instances = []

        for gold_sentence, predicted_sentence in zip(gold, predicted):
            assert (gold_sentence.original_text ==
                    predicted_sentence.original_text), (
                        "Can't compare annotations on non-identical sentences")
            gold_causations = self.__get_causations(gold_sentence, True)
            predicted_causations = self.__get_causations(predicted_sentence, False)
            
            sentence_matching, sentence_gold_only, sentence_predicted_only = (
                self.get_connective_matches(
                    gold_causations, predicted_causations, self.allow_partial))
            matching_instances.extend(sentence_matching)
            gold_only_instances.extend(sentence_gold_only)
            predicted_only_instances.extend(sentence_predicted_only)

        if self.ids_considered == CausalityMetrics.IDsConsidered.GivenOnly:
            assert len(matching_instances) == len(
                FLAGS.iaa_given_connective_ids), (
                    "Didn't find all expected given connectives! Perhaps"
                    " annotators re-annotated spans with different IDs?")
            # Leave connective_metrics as None to indicate that there aren't
            # any interesting values here. (Everything should be perfect.)
            connective_metrics = None
        # "Both" will only affect the connective stats if there are actually
        # some given connectives.
        elif (self.ids_considered == CausalityMetrics.IDsConsidered.Both
              and FLAGS.iaa_given_connective_ids):
            connective_metrics = None
        else:
            connective_metrics = ClassificationMetrics(
                len(matching_instances), len(predicted_only_instances),
                len(gold_only_instances))

        def sentences_by_file(sentences):
            by_file = defaultdict(list)
            for sentence in sentences:
                filename = os.path.split(sentence.source_file_path)[-1]
                by_file[filename].append(sentence)
            return by_file

        if self.save_differences or self.save_agreements:
            gold_by_file = sentences_by_file(gold)

        if self.save_differences:
            predicted_by_file = sentences_by_file(predicted)
            self.gold_only_instances = [
                (gold_by_file[os.path.split(i.sentence.source_file_path)[-1]]
                 .index(i.sentence) + 1, i)
                for i in gold_only_instances]
            self.predicted_only_instances = [
                (predicted_by_file[os.path.split(
                    i.sentence.source_file_path)[-1]]
                 .index(i.sentence) + 1, i)
                for i in predicted_only_instances]
            
        if self.save_agreements:
            self.agreeing_instances = [
                (gold_by_file[os.path.split(i1.sentence.source_file_path)[-1]]
                 .index(i1.sentence) + 1, i1)
                for i1, i2 in matching_instances]

        return (connective_metrics, matching_instances)
    
    def _get_jaccard(self, matches, arg_property_name):
        '''
        Returns average Jaccard index across `matches` for property
        `arg_property_name` (cause or effect).
        '''
        jaccard_avg_numerator = 0
        def get_arg_indices(instance):
            arg = getattr(instance, arg_property_name)
            if arg is None:
                return []
            else:
                if not FLAGS.iaa_check_punct:
                    arg = self._filter_punct_tokens(arg)
                return [token.index for token in arg]

        for instance_pair in matches:
            i1_indices, i2_indices = [get_arg_indices(i) for i in instance_pair]
            if i1_indices or i2_indices:
                # TODO: This could maybe be done more efficiently by walking
                # along each set of indices.
                diff = SequenceDiff(i1_indices, i2_indices)
                num_matching = len(diff.get_matching_pairs())
                match_jaccard = num_matching / float(
                    len(i1_indices) + len(i2_indices) - num_matching)
            else:
                match_jaccard = 1.0
            jaccard_avg_numerator += match_jaccard

        return jaccard_avg_numerator / safe_divisor(float(len(matches)))

    def _compute_agreement_matrix(self, matches, labels_enum, property_name,
                                  gold_sentences):
        labels_1 = []
        labels_2 = []

        def log_missing(instance, number):
            print(property_type_name,
                  ('property not set in Annotation %d;' % number),
                  'not including in analysis (sentence: "',
                  _wrapped_sentence_highlighting_instance(instance_1).encode(
                      'utf-8') + '")',
                  file=sys.stderr)

        for instance_1, instance_2 in matches:
            property_1 = getattr(instance_1, property_name)
            property_2 = getattr(instance_2, property_name)

            property_type_name = (["Degree", "Causation type"]
                                  [property_name == 'type'])
            if property_1 >= len(labels_enum):
                log_missing(instance_1, 1)
            elif property_2 >= len(labels_enum):
                log_missing(instance_2, 2)
            else:
                labels_1.append(labels_enum[property_1])
                labels_2.append(labels_enum[property_2])
                sentence_num = gold_sentences.index(instance_1.sentence) + 1
                if property_1 != property_2 and self.save_differences:
                    self.property_differences.append(
                        (instance_1, instance_2, labels_enum, sentence_num))

        return ConfusionMatrix(labels_1, labels_2)

    def _match_instance_args(self, arg_1, arg_2):
        if arg_1 is None or arg_2 is None:
            if arg_1 is arg_2: # both None
                spans_match = True
                heads_match = True
            else: # one is None and the other isn't
                spans_match = False
                heads_match = False
        else:
            spans_match = self._annotation_comparator(arg_1, arg_2)
            arg_1_head, arg_2_head = [arg[0].parent_sentence.get_head(arg)
                                      for arg in [arg_1, arg_2]]
            if arg_1_head is None or arg_2_head is None:
                if arg_1_head is arg_2_head: # both None
                    heads_match = True
                else:
                    heads_match = False
            else:
                heads_match = (arg_1_head.index == arg_2_head.index)

        return spans_match, heads_match
    
    @staticmethod
    def _filter_punct_tokens(tokens):
        return [t for t in tokens if t.pos not in Token.PUNCT_TAGS]

    def _match_arguments(self, matches, gold):
        correct_causes = 0
        correct_effects = 0

        correct_cause_heads = 0
        correct_effect_heads = 0

        for instance_1, instance_2 in matches:
            sentence_num = gold.index(instance_1.sentence) + 1
            if FLAGS.iaa_check_punct:
                cause_1, cause_2 = [i.cause for i in [instance_1, instance_2]]
                effect_1, effect_2 = [i.effect for i
                                      in [instance_1, instance_2]]
            else:
                cause_1, cause_2 = [self._filter_punct_tokens(i.cause)
                                    if i.cause else None
                                    for i in [instance_1, instance_2]]
                effect_1, effect_2 = [self._filter_punct_tokens(i.effect)
                                      if i.effect else None
                                      for i in [instance_1, instance_2]]

            causes_match, cause_heads_match = self._match_instance_args(
                cause_1, cause_2)
            effects_match, effect_heads_match = self._match_instance_args(
                effect_1, effect_2)
            correct_causes += causes_match
            correct_effects += effects_match
            correct_cause_heads += cause_heads_match
            correct_effect_heads += effect_heads_match

            # If there's any difference, record it.
            if (self.save_differences and
                not (causes_match and effects_match and
                     cause_heads_match and effect_heads_match)):
                self.argument_differences.append((instance_1, instance_2,
                                                  sentence_num))

        cause_span_metrics = AccuracyMetrics(correct_causes,
                                             len(matches) - correct_causes)
        effect_span_metrics = AccuracyMetrics(correct_effects,
                                              len(matches) - correct_effects)
        cause_head_metrics = AccuracyMetrics(
            correct_cause_heads, len(matches) - correct_cause_heads)
        effect_head_metrics = AccuracyMetrics(
            correct_effect_heads, len(matches) - correct_effect_heads)

        return (cause_span_metrics, effect_span_metrics,
                cause_head_metrics, effect_head_metrics)

    def pp(self, log_confusion=None, log_stats=None, log_differences=None,
           log_agreements=None, indent=0, file=sys.stdout):
        # Flags aren't available as defaults when the function is created, so
        # set the defaults here.
        if log_confusion is None:
            log_confusion = FLAGS.iaa_log_confusion
        if log_stats is None:
            log_stats = FLAGS.iaa_log_stats
        if log_differences is None:
            log_differences = FLAGS.iaa_log_differences
        if log_agreements is None:
            log_agreements = FLAGS.iaa_log_agreements

        if log_differences:
            colorama.reinit()

        if log_agreements:
            print_indented(indent, "Agreeing instances:", file=file)
            for sentence_num, instance in self.agreeing_instances:
                self._log_instance_for_connective(
                    instance, sentence_num, "", indent + 1, file)

        if log_differences and (
            self.gold_only_instances or self.predicted_only_instances
            or self.property_differences or self.argument_differences):
            print_indented(indent, 'Annotation differences:', file=file)
            for sentence_num, instance in self.gold_only_instances:
                self._log_instance_for_connective(
                    instance, sentence_num, "Annotator 1 only:", indent + 1,
                    file)
            for sentence_num, instance in self.predicted_only_instances:
                self._log_instance_for_connective(
                    instance, sentence_num, "Annotator 2 only:", indent + 1,
                    file)
            self._log_property_differences(CausationInstance.CausationTypes,
                                           indent + 1, file)
            self._log_property_differences(CausationInstance.Degrees,
                                           indent + 1, file)
            self._log_arg_label_differences(indent + 1, file)

        # Ignore connective-related metrics if we have nothing interesting to
        # show there.
        printing_connective_metrics = (log_stats and self.connective_metrics)
        if printing_connective_metrics or log_confusion:
            print_indented(indent, 'Connectives:', file=file)
        if printing_connective_metrics:
            print_indented(indent + 1, self.connective_metrics, file=file)
        if log_stats or log_confusion:
            if self.degree_matrix is not None:
                self._log_property_metrics(
                    'Degrees', self.degree_matrix, indent + 1, log_confusion,
                    log_stats, file)
            if self.causation_type_matrix is not None:
                self._log_property_metrics(
                    'Causation types', self.causation_type_matrix, indent + 1,
                    log_confusion, log_stats, file)

        # If any argument properties are set, all should be.
        if log_stats and self.cause_span_metrics is not None:
            print_indented(indent, 'Arguments:', file=file)
            for arg_type in ['cause', 'effect']:
                print_indented(indent + 1, arg_type.title(), 's:', sep='',
                               file=file)
                print_indented(indent + 2, 'Spans:', file=file)
                print_indented(indent + 3, getattr(self,
                                                   arg_type + '_span_metrics'),
                               file=file)
                print_indented(indent + 2, 'Heads:', file=file)
                print_indented(indent + 3, getattr(self,
                                                   arg_type + '_head_metrics'),
                               file=file)
                print_indented(indent + 2, 'Jaccard index: ',
                               getattr(self, arg_type + '_jaccard'), file=file)

        if log_differences:
            colorama.deinit()

    def __repr__(self):
        '''
        This is a dumb hack, but it's easier than trying to rewrite all of pp to
        operate on strings, and possibly faster too (since then we'd have to
        keep copying strings over to concatenate them).
        '''
        string_buffer = StringIO()
        self.pp(None, None, None, None, 0, string_buffer)
        return string_buffer.getvalue()

    @staticmethod
    def aggregate(metrics_list):
        '''
        Aggregates IAA statistics. Classification and accuracy metrics are
        averaged; confusion matrices are summed.
        '''
        assert metrics_list, "Can't aggregate empty list of causality metrics!"
        aggregated = object.__new__(CausalityMetrics)
        # For an aggregated, it won't make sense to list all the individual
        # sets of instances/properties processed in the individual computations.
        for attr_name in [
            'ids_considered', 'gold_only_instances',
            'predicted_only_instances', 'property_differences',
            'argument_differences', 'agreeing_instances']:
            setattr(aggregated, attr_name, [])
        aggregated.save_differences = None

        aggregated.connective_metrics = ClassificationMetrics.average(
            [m.connective_metrics for m in metrics_list])
        degrees = [m.degree_matrix for m in metrics_list]
        if None not in degrees:
            aggregated.degree_matrix = reduce(operator.add, degrees)
        else:
            aggregated.degree_matrix = None
        causation_types = [m.causation_type_matrix for m in metrics_list]
        if None not in causation_types:
            aggregated.causation_type_matrix = reduce(operator.add,
                                                      causation_types)
        else:
            aggregated.causation_type_matrix = None

        for attr_name in ['cause_span_metrics', 'effect_span_metrics',
                          'cause_head_metrics', 'effect_head_metrics']:
            attr_values = [getattr(m, attr_name) for m in metrics_list]
            attr_values = [v for v in attr_values if v is not None]
            if attr_values:
                setattr(aggregated, attr_name,
                        AccuracyMetrics.average(attr_values))
            else:
                setattr(aggregated, attr_name, None)

        for attr_name in ['cause_jaccard', 'effect_jaccard']:
            attr_values = [getattr(m, attr_name) for m in metrics_list]
            attr_values = [v for v in attr_values if v is not None]
            if attr_values: # At least some are not None
                attr_values = [v for v in attr_values if not np.isnan(v)]
                if attr_values:
                    aggregate_value = np.mean(attr_values)
                else:
                    aggregate_value = np.nan
            else:
                aggregate_value = None

            setattr(aggregated, attr_name, aggregate_value)

        return aggregated

    def _log_property_metrics(self, name, matrix, indent, log_confusion,
                              log_stats, file):
        print_indented(indent, name, ':', sep='', file=file)
        if log_confusion:
            print_indented(indent + 1, matrix.pretty_format(metrics=log_stats),
                           file=file)
        else: # we must be logging just stats
            print_indented(indent + 1, matrix.pretty_format_metrics(),
                           file=file)

    @staticmethod
    def _log_instance_for_connective(instance, sentence_num, msg, indent, file):
        filename = os.path.split(instance.sentence.source_file_path)[-1]
        print_indented(
            indent, msg,
            _wrapped_sentence_highlighting_instance(instance).encode('utf-8'),
            '(%s:%d)' % (filename, sentence_num),
            file=file)

    @staticmethod
    def _print_with_labeled_args(instance, indent, file, cause_start, cause_end,
                                 effect_start, effect_end):
        '''
        Prints sentences annotated according to a particular CausationInstance.
        Connectives are printed in ALL CAPS. If run from a TTY, arguments are
        printed in color; otherwise, they're indicated as '/cause/' and
        '*effect*'. 
        '''
        def get_printable_word(token):
            word = token.original_text
            if token in instance.connective:
                word = _get_printable_connective_word(word)

            if instance.cause and token in instance.cause:
                word = cause_start + word + cause_end
            elif instance.effect and token in instance.effect:
                word = effect_start + word + effect_end
            return word
            
        tokens = instance.sentence.tokens[1:] # skip ROOT
        if sys.stdout.isatty() or FLAGS.iaa_force_color:
            words = [token.original_text
                     for token in tokens]
            # -10 allows viewing later in a slightly smaller terminal/editor.
            available_term_width = get_terminal_size()[0] - indent * 4 - 10
        else:
            words = [get_printable_word(token) for token in tokens]
            available_term_width = 75 - indent * 4 # 75 to allow for long words
        lines = wrap(' '.join(words), available_term_width,
                     subsequent_indent='    ', break_long_words=False)

        # For TTY, we now have to re-process the lines to add in color and
        # capitalizations.
        if sys.stdout.isatty() or FLAGS.iaa_force_color:
            tokens_processed = 0
            for i, line in enumerate(lines):
                # NOTE: This assumes no tokens with spaces in them.
                words = line.split()
                zipped = zip(words, tokens[tokens_processed:])
                printable_line = ' '.join([get_printable_word(token)
                                           for _, token in zipped])
                print_indented(indent, printable_line.encode('utf-8'))
                tokens_processed += len(words)
                if i == 0:
                    indent += 1 # future lines should be printed more indented
        else: # non-TTY: we're ready to print
            print_indented(indent, *[line.encode('utf-8') for line in lines],
                           sep='\n')

    def _log_arg_label_differences(self, indent, file):
        if sys.stdout.isatty() or FLAGS.iaa_force_color:
            cause_start = getattr(colorama.Fore, FLAGS.iaa_cause_color.upper())
            cause_end = colorama.Fore.RESET
            effect_start = getattr(colorama.Fore, FLAGS.iaa_effect_color.upper())
            effect_end = colorama.Fore.RESET
        else:
            cause_start = '/'
            cause_end = '/'
            effect_start = '*'
            effect_end = '*'
        
        for instance_1, instance_2, sentence_num in self.argument_differences:
            filename = os.path.split(instance_1.sentence.source_file_path)[-1]
            connective_text = StanfordParsedSentence.get_annotation_text(
                    instance_1.connective).encode('utf-8)')
            print_indented(
                indent,
                'Arguments differ for connective "', connective_text,
                '" (', filename, ':', sentence_num, ')',
                ' with ', cause_start, 'cause', cause_end, ' and ',
                effect_start, 'effect', effect_end, ':',
                sep='', file=file)
            self._print_with_labeled_args(
                instance_1, indent + 1, file, cause_start, cause_end,
                effect_start, effect_end)
            # print_indented(indent + 1, "vs.", file=file)
            self._print_with_labeled_args(
                instance_2, indent + 1, file, cause_start, cause_end,
                effect_start, effect_end)

    def _log_property_differences(self, property_enum, indent, file):
        filtered_differences = [x for x in self.property_differences
                                if x[2] is property_enum]

        if property_enum is CausationInstance.Degrees:
            property_name = 'Degree'
            value_extractor = lambda instance: instance.Degrees[instance.degree]
        elif property_enum is CausationInstance.CausationTypes:
            property_name = 'Causation type'
            value_extractor = lambda instance: (
                instance.CausationTypes[instance.type])

        for instance_1, instance_2, _, sentence_num in filtered_differences:
            if value_extractor:
                values = (value_extractor(instance_1),
                          value_extractor(instance_2))
            filename = os.path.split(instance_1.sentence.source_file_path)[-1]
            print_indented(
                indent, property_name, 's for connective "',
                StanfordParsedSentence.get_annotation_text(
                    instance_1.connective).encode('utf-8)'),
                '" differ: ', values[0], ' vs. ', values[1],
                ' (', filename, ':', sentence_num, ': "',
                _wrapped_sentence_highlighting_instance(instance_1), '")',
                sep='', file=file)
