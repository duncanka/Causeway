from __future__ import print_function

from collections import defaultdict
import colorama
import itertools
colorama.init()
colorama.deinit()
from copy import copy
from StringIO import StringIO
from gflags import (DEFINE_list, DEFINE_float, DEFINE_bool, DEFINE_string,
                    DuplicateFlagError, FLAGS)
import logging
import numpy as np
import operator
import os
import sys
from textwrap import wrap

from causeway.because_data import CausationInstance, OverlappingRelationInstance
from nlpypline.data import StanfordParsedSentence, Token
from nlpypline.util import (Enum, print_indented, truncated_string,
                            get_terminal_size, merge_dicts, make_setter,
                            make_getter)
from nlpypline.util.diff import SequenceDiff
from nlpypline.util.metrics import (ClassificationMetrics, ConfusionMatrix,
                                    AccuracyMetrics, safe_divide,
                                    FloatWithStddev)

np.seterr(divide='ignore') # Ignore nans in division

try:
    DEFINE_list(
        'iaa_given_connective_ids', [], "Annotation IDs for connectives"
        " that were given as gold and should be treated separately for IAA.")
    DEFINE_float(
        'iaa_min_partial_overlap', 0.25, "Minimum fraction of the larger of two"
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
    DEFINE_string('iaa_means_color', 'Magenta',
                  'ANSI color to use for formatting means words in IAA'
                  ' comparison output')
    DEFINE_bool('iaa_force_color', False,
                "Force ANSI color in IAA comparisons even when we're not"
                " outputting to a TTY")
    DEFINE_bool('iaa_check_punct', False,
                'Whether IAA should compare punctuation tokens to determine'
                ' argument matches')
    DEFINE_bool('iaa_log_by_connective', False,
                "When logging a stage's results, include per-connective stats")
    DEFINE_bool('iaa_log_by_category', False,
                "When logging a stage's results, include per-category stats")
    DEFINE_bool('iaa_compute_overlapping', True,
                "Compute overlapping relations as part of causality IAA stats")
except DuplicateFlagError as e:
    logging.warn('Ignoring flag redefinitions; assuming module reload')


def make_annotation_comparator(allow_partial):
    def match_annotations(token_list_1, token_list_2):
        if allow_partial:
            min_partial_overlap = FLAGS.iaa_min_partial_overlap
        else:
            min_partial_overlap = 1.0

        # Just in case we accidentally added tokens to an annotation in the
        # wrong order, sort by token position.
        sort_key = lambda token: (token.parent_sentence.document_char_offset,
                                  token.index)
        offsets_1 = [(token.start_offset, token.end_offset)
                     for token in sorted(token_list_1, key=sort_key)]
        offsets_2 = [(token.start_offset, token.end_offset)
                     for token in sorted(token_list_2, key=sort_key)]
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
        return colorama.Style.BRIGHT + word.upper() + colorama.Style.RESET_ALL
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


class ArgumentMetrics(object):
    def __init__(self, span_metrics, head_metrics, jaccard):
        self.span_metrics = span_metrics
        self.head_metrics = head_metrics
        self.jaccard = jaccard

    def __add__(self, other):
        if self.span_metrics is not None and other.span_metrics is not None:
            added_span_metrics = self.span_metrics + other.span_metrics
        else:
            added_span_metrics = None
        if self.head_metrics is not None and other.head_metrics is not None:
            added_head_metrics = self.head_metrics + other.head_metrics
        else:
            added_head_metrics = None

        # Ignore NaNs and Nones.
        if self.jaccard is None or np.isnan(self.jaccard):
            added_jaccard = other.jaccard
        elif other.jaccard is None or np.isnan(other.jaccard):
            added_jaccard = self.jaccard
        else:
            added_jaccard = (self.jaccard + other.jaccard) / 2.0
        return ArgumentMetrics(added_span_metrics, added_head_metrics,
                               added_jaccard)


class _RelationMetrics(object):
    IDsConsidered = Enum(['GivenOnly', 'NonGivenOnly', 'Both'])
    _SAVED_ATTR_NAMES = ['gold_only_instances', 'predicted_only_instances',
                          'agreeing_instances', 'property_differences',
                          'argument_differences']
    # To be overridden by subclasses
    _GOLD_INSTANCES_PROPERTY_NAME = None
    _INSTANCE_CLASS = None

    # TODO: Refactor order of parameters
    def __init__(self, gold, predicted, allow_partial, save_differences,
                 ids_considered, compare_args, properties_to_compare,
                 pairwise_only, save_agreements, instances_property_name):
        # properties_to_compare is a list of (property_name,
        # property_values_enum, should_compare_property) tuples.
        assert len(gold) == len(predicted), (
            "Cannot compute IAA for different-sized datasets")

        if ids_considered is None:
            ids_considered = _RelationMetrics.IDsConsidered.Both
        self.allow_partial = allow_partial
        self._annotation_comparator = make_annotation_comparator(allow_partial)
        self.ids_considered = ids_considered
        self.save_differences = save_differences
        self.save_agreements = save_agreements
        self.pairwise_only = pairwise_only
        self.instances_property_name = instances_property_name
        self.properties_to_compare = properties_to_compare

        self.gold_only_instances = []
        self.predicted_only_instances = []
        self.agreeing_instances = []
        self.argument_differences = []
        self.property_differences = []

        # Compute attributes that take a little more work.
        self.connective_metrics, matches = self._match_connectives(
            gold, predicted)

        for property_name, property_enum, compare_property in (
                properties_to_compare):
            matrix_attr_name = '%s_matrix' % property_name
            if compare_property:
                matrix = self._compute_agreement_matrix(matches, property_enum,
                                                        property_name, gold)
                setattr(self, matrix_attr_name, matrix)
            else:
                setattr(self, matrix_attr_name, None)

        # TODO: add back metrics that account for the possibility of null args
        # (i.e., P/R/F1).
        if compare_args:
            self._match_arguments(matches, gold)
        else:
            null_metrics = ArgumentMetrics(None, None, None)
            for arg_num in range(self._INSTANCE_CLASS._num_args):
                setattr(self, 'arg%d_metrics' % arg_num, null_metrics)

    def __add__(self, other):
        if (self.allow_partial != other.allow_partial or
            self.properties_to_compare != other.properties_to_compare):
            raise ValueError("Can't add binary relation annotation metrics with"
                             " different comparison criteria")

        sum_metrics = copy(self)
        # Add recorded instances/differences.
        for attr_name in self._SAVED_ATTR_NAMES:
            getattr(sum_metrics, attr_name).extend(getattr(other, attr_name))

        # Add together submetrics, if they exist
        sum_metrics.connective_metrics += other.connective_metrics
        submetric_names = ['_'.join([property_name, 'matrix']) for
                           (property_name, _, _) in self.properties_to_compare]
        submetric_names += ['%s_metrics' % arg_type for arg_type
                            in self._INSTANCE_CLASS.get_arg_types()]
        for attr_name in submetric_names:
            self_attr = getattr(self, attr_name)
            other_attr = getattr(other, attr_name)
            if self_attr is not None and other_attr is not None:
                attr_value = self_attr + other_attr
            else:
                attr_value = None
            setattr(sum_metrics, attr_name, attr_value)

        return sum_metrics

    def __get_instances(self, sentence, is_gold):
        instances = []
        property_name = [self.instances_property_name,
                         self._GOLD_INSTANCES_PROPERTY_NAME][is_gold]
        for instance in getattr(sentence, property_name):
            if (# First set of conditions: matches givenness specified
                (self.ids_considered == self.IDsConsidered.Both or
                 (instance.id in FLAGS.iaa_given_connective_ids and
                  self.ids_considered == self.IDsConsidered.GivenOnly) or
                 (instance.id not in FLAGS.iaa_given_connective_ids and
                  self.ids_considered == self.IDsConsidered.NonGivenOnly))
                # Second set of conditions: is pairwise if necessary
                and (not is_gold or not self.pairwise_only or
                     (instance.arg0 != None and instance.arg1 != None))):
                instances.append(instance)
        return instances

    @staticmethod
    def get_connective_matches(gold_instances, predicted_instances,
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
        # head, or what happens if arg0 is None?
        sort_key = lambda inst: (
            inst.connective[0].start_offset,
            inst.arg0[0].start_offset if inst.arg0 else 0,
            inst.arg1[0].start_offset if inst.arg1 else 0)

        matching_instances = []
        gold_only_instances = []
        predicted_only_instances = []

        # If we're allowing partial matches, we don't want any partial
        # matches to override full matches. So we first do an exact match,
        # and remove the ones that matched from the partial matching.
        if allow_partial:
            diff = SequenceDiff(gold_instances, predicted_instances,
                                compare_connectives_exact, sort_key)
            matching_pairs = diff.get_matching_pairs()
            matching_instances.extend(matching_pairs)
            # Instances that were gold-only or predicted-only may still generate
            # partial matches.
            gold_instances = diff.get_a_only_elements()
            predicted_instances = diff.get_b_only_elements()

        diff = SequenceDiff(gold_instances, predicted_instances,
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
            gold_instances = self.__get_instances(gold_sentence, True)
            predicted_instances = self.__get_instances(predicted_sentence,
                                                       False)

            sentence_matching, sentence_gold_only, sentence_predicted_only = (
                self.get_connective_matches(
                    gold_instances, predicted_instances, self.allow_partial))
            matching_instances.extend(sentence_matching)
            gold_only_instances.extend(sentence_gold_only)
            predicted_only_instances.extend(sentence_predicted_only)

        if (self.ids_considered ==
            _RelationMetrics.IDsConsidered.GivenOnly):
            assert len(matching_instances) == len(
                FLAGS.iaa_given_connective_ids), (
                    "Didn't find all expected given connectives! Perhaps"
                    " annotators re-annotated spans with different IDs?")
            # Leave connective_metrics as None to indicate that there aren't
            # any interesting values here. (Everything should be perfect.)
            connective_metrics = None
        # "Both" will only affect the connective stats if there are actually
        # some given connectives.
        elif (self.ids_considered == _RelationMetrics.IDsConsidered.Both
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
                for i1, _i2 in matching_instances]

        return (connective_metrics, matching_instances)

    def _get_jaccard(self, matches, arg_property_name):
        '''
        Returns average Jaccard index across `matches` for property
        `arg_property_name`.
        '''
        iaa_check_punct = FLAGS.iaa_check_punct
        jaccard_avg_numerator = 0
        def get_arg_indices(instance):
            arg = getattr(instance, arg_property_name)
            if arg is None:
                return []
            else:
                if not iaa_check_punct:
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
            else: # both empty arguments; overlap is defined as 1.
                match_jaccard = 1.0
            jaccard_avg_numerator += match_jaccard

        return safe_divide(jaccard_avg_numerator, len(matches))

    def _compute_agreement_matrix(self, matches, labels_enum, property_name,
                                  gold_sentences):
        labels_1 = []
        labels_2 = []

        def log_missing(instance, number):
            print(property_name,
                  ('property not set in Annotation %d;' % number),
                  'not including in analysis (sentence: "',
                  _wrapped_sentence_highlighting_instance(instance_1).encode(
                      'utf-8') + '")',
                  file=sys.stderr)

        for instance_1, instance_2 in matches:
            property_1 = getattr(instance_1, property_name)
            property_2 = getattr(instance_2, property_name)

            if property_1 >= len(labels_enum) or property_1 is None:
                log_missing(instance_1, 1)
            elif property_2 >= len(labels_enum) or property_2 is None:
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

    def _match_arguments(self, matches, gold=None):
        iaa_check_punct = FLAGS.iaa_check_punct
        correct_args = [0] * self._INSTANCE_CLASS._num_args
        correct_heads = [0] * self._INSTANCE_CLASS._num_args

        for instance_1, instance_2 in matches:
            if gold is not None:
                sentence_num = gold.index(instance_1.sentence) + 1
            else:
                sentence_num = -1 # No valid sentence number

            i1_args = instance_1.get_args()
            i2_args = instance_2.get_args()
            all_args_match = True
            for arg_num in range(self._INSTANCE_CLASS._num_args):
                if iaa_check_punct:
                    first_arg, second_arg = i1_args[arg_num], i2_args[arg_num]
                else:
                    first_arg, second_arg = [
                        self._filter_punct_tokens(arg) if arg else None
                        for arg in [i1_args[arg_num], i2_args[arg_num]]]

                args_match, arg_heads_match = self._match_instance_args(
                    first_arg, second_arg)
                correct_args[arg_num] += args_match
                correct_heads[arg_num] += arg_heads_match
                all_args_match = all_args_match and args_match

            # If there's any difference, record it.
            if self.save_differences and not all_args_match:
                self.argument_differences.append((instance_1, instance_2,
                                                  sentence_num))

        for arg_num in range(self._INSTANCE_CLASS._num_args):
            arg_name = 'arg%d' % arg_num
            arg_jaccard = self._get_jaccard(matches, arg_name)
            arg_span_metrics = AccuracyMetrics(
                correct_args[arg_num], len(matches) - correct_args[arg_num])
            arg_head_metrics = AccuracyMetrics(
                correct_heads[arg_num], len(matches) - correct_heads[arg_num])
            arg_metrics = ArgumentMetrics(arg_span_metrics, arg_head_metrics,
                                          arg_jaccard)
            setattr(self, arg_name + '_metrics', arg_metrics)

    def pp(self, log_confusion=None, log_stats=None, log_differences=None,
           log_agreements=None, log_by_connective=None, indent=0,
           log_file=sys.stdout):
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
        if log_by_connective is None:
            log_by_connective = FLAGS.iaa_log_by_connective

        if log_differences:
            colorama.reinit()

        if log_agreements:
            print_indented(indent, "Agreeing instances:", file=log_file)
            for sentence_num, instance in self.agreeing_instances:
                self._log_instance_for_connective(
                    instance, sentence_num, "", indent + 1, log_file)

        if log_differences and (
            self.gold_only_instances or self.predicted_only_instances
            or self.property_differences or self.argument_differences):
            print_indented(indent, 'Annotation differences:', file=log_file)
            for sentence_num, instance in self.gold_only_instances:
                self._log_instance_for_connective(
                    instance, sentence_num, "Annotator 1 only:", indent + 1,
                    log_file)
            for sentence_num, instance in self.predicted_only_instances:
                self._log_instance_for_connective(
                    instance, sentence_num, "Annotator 2 only:", indent + 1,
                    log_file)

            for property_name, property_enum, comparing_property in (
                self.properties_to_compare):
                if comparing_property:
                    self._log_property_differences(property_name, property_enum,
                                                   indent + 1, log_file)

            self._log_arg_label_differences(indent + 1, log_file)

        # Ignore connective-related metrics if we have nothing interesting to
        # show there.
        printing_connective_metrics = (log_stats and self.connective_metrics)
        if printing_connective_metrics or log_confusion:
            print_indented(indent, 'Connectives:', file=log_file)
        if printing_connective_metrics:
            print_indented(indent + 1, self.connective_metrics, file=log_file)
        if log_stats or log_confusion:
            for property_name, property_enum, comparing_property in (
                self.properties_to_compare):
                if comparing_property:
                    matrix_attr_name = '%s_matrix' % property_name
                    self._log_property_metrics(
                        property_name, getattr(self, matrix_attr_name),
                        indent + 1, log_confusion, log_stats, log_file)

        # If any argument properties are set, all should be.
        if log_stats and self.arg0_metrics is not None:
            print_indented(indent, 'Arguments:', file=log_file)
            for arg_type in self._INSTANCE_CLASS.get_arg_types():
                arg_name = self._INSTANCE_CLASS.arg_names[arg_type]
                print_indented(indent + 1, arg_name.title(),
                               's:' if arg_name[-1] != 's' else ':',
                               sep='', file=log_file)
                print_indented(indent + 2, 'Spans:', file=log_file)
                print_indented(
                    indent + 3,
                    getattr(self, arg_type + '_metrics').span_metrics,
                    file=log_file)
                print_indented(indent + 2, 'Heads:', file=log_file)
                print_indented(
                    indent + 3,
                    getattr(self, arg_type + '_metrics').head_metrics,
                    file=log_file)
                print_indented(indent + 2, 'Jaccard index: ',
                               getattr(self, arg_type + '_metrics').jaccard,
                               file=log_file)

        if log_differences:
            colorama.deinit()

        if log_by_connective and any(getattr(self, attr_name)
                                     for attr_name in self._SAVED_ATTR_NAMES):
            print(file=log_file)
            print_indented(indent, 'Metrics by connective:', file=log_file)
            by_connective = self.metrics_by_connective()
            print_indented(indent + 1, self._csv_metrics(by_connective),
                           file=log_file)

    def metrics_by_connective(self):
        return self.get_aggregate_metrics(stringify_connective)

    def get_aggregate_metrics(self, instance_to_category):
        metrics = defaultdict(lambda: type(self)([], [], False))

        # Compute connective accuracy metrics by category.
        for _, instance in self.agreeing_instances:
            metrics[instance_to_category(instance)].connective_metrics.tp += 1
        for _, instance in self.gold_only_instances:
            metrics[instance_to_category(instance)].connective_metrics.fn += 1
        for _, instance in self.predicted_only_instances:
            metrics[instance_to_category(instance)].connective_metrics.fp += 1

        for category_metrics in metrics.values():
            category_metrics.connective_metrics._finalize_counts()

        # Compute arg match metrics by connective.
        arg_diffs_by_category = defaultdict(list)
        for instance_1, instance_2, _ in self.argument_differences:
            connective = instance_to_category(instance_1)
            arg_diffs_by_category[connective].append((instance_1, instance_2))

        for connective, conn_matches in arg_diffs_by_category.iteritems():
            conn_metrics = metrics[connective]
            if self.arg0_metrics is not None: # we're matching arguments
                conn_metrics._match_arguments(conn_matches)
            else:
                null_metrics = ArgumentMetrics(None, None, None)
                for arg_num in range(self._INSTANCE_CLASS._num_args):
                    setattr(conn_metrics, 'arg%d_metrics' % arg_num,
                            null_metrics)

        return metrics

    def __repr__(self):
        '''
        This is a dumb hack, but it's easier than trying to rewrite all of pp to
        operate on strings, and possibly faster too (since then we'd have to
        keep copying strings over to concatenate them).
        '''
        string_buffer = StringIO()
        self.pp(indent=0, log_file=string_buffer)
        return string_buffer.getvalue()

    @staticmethod
    def aggregate(metrics_list):
        '''
        Aggregates IAA statistics. Classification and accuracy metrics are
        averaged; confusion matrices are summed.
        '''
        assert metrics_list, "Can't aggregate empty list of metrics!"
        metrics_type = type(metrics_list[0])
        aggregated = object.__new__(metrics_type)

        aggregated.ids_considered = None
        aggregated.save_differences = any(m.save_differences
                                          for m in metrics_list)
        # TODO: confirm that all properties_to_compare lists are the same?
        aggregated.properties_to_compare = metrics_list[0].properties_to_compare
        # Save lists of instances needed for metrics_by_connective.
        for attr_name in metrics_type._SAVED_ATTR_NAMES:
            if aggregated.save_differences:
                all_relevant_instances = itertools.chain.from_iterable(
                    getattr(m, attr_name) for m in metrics_list)
                setattr(aggregated, attr_name, list(all_relevant_instances))
            else:
                setattr(aggregated, attr_name, [])

        aggregated.connective_metrics = (
            metrics_list[0].connective_metrics.average(
                [m.connective_metrics for m in metrics_list]))

        property_matrices = defaultdict(list)
        for m in metrics_list:
            for property_name, _property_enum, compare_property in (
                m.properties_to_compare):
                matrix_attr_name = '%s_matrix' % property_name
                if not compare_property:
                    property_matrices[matrix_attr_name] = None
                else:
                    matrices = property_matrices[matrix_attr_name]
                    if matrices is not None:
                        matrices.append(getattr(m, matrix_attr_name))

        for matrix_name, matrices in property_matrices.iteritems():
            try:
                aggregated_matrix = reduce(operator.add, matrices)
            except TypeError: # happens if matrices is None
                aggregated_matrix = None
            setattr(aggregated, matrix_name, aggregated_matrix)

        for arg_type in metrics_type._INSTANCE_CLASS.get_arg_types():
            arg_metrics_attr_name = arg_type + '_metrics'
            arg_metrics = object.__new__(ArgumentMetrics)
            setattr(aggregated, arg_metrics_attr_name, arg_metrics)

            for sub_attr_name in ['span_metrics', 'head_metrics']:
                sub_attr_values = [getattr(getattr(m, arg_metrics_attr_name),
                                       sub_attr_name) for m in metrics_list]
                sub_attr_values = [v for v in sub_attr_values if v is not None]
                if sub_attr_values:
                    setattr(arg_metrics, sub_attr_name,
                            sub_attr_values[0].average(sub_attr_values))
                else:
                    setattr(arg_metrics, sub_attr_name, None)

            jaccard_values = [getattr(m, arg_metrics_attr_name).jaccard
                              for m in metrics_list]
            jaccard_values = [v for v in jaccard_values if v is not None]
            if jaccard_values: # At least some are not None
                jaccard_values = [v for v in jaccard_values if not np.isnan(v)]
                if jaccard_values:
                    arg_metrics.jaccard = FloatWithStddev.from_list(
                        jaccard_values)
                else:
                    arg_metrics.jaccard = np.nan
            else:
                arg_metrics.jaccard = None

        return aggregated

    def _log_property_metrics(self, name, matrix, indent, log_confusion,
                              log_stats, log_file):
        print_name = name.title() + 's'
        print_indented(indent, print_name, ':', sep='', file=log_file)
        if log_confusion:
            print_indented(indent + 1, matrix.pretty_format(metrics=log_stats),
                           file=log_file)
        else: # we must be logging just stats
            print_indented(indent + 1, matrix.pretty_format_metrics(),
                           file=log_file)

    @staticmethod
    def _log_instance_for_connective(instance, sentence_num, msg, indent,
                                     log_file):
        filename = os.path.split(instance.sentence.source_file_path)[-1]
        print_indented(
            indent, msg,
            _wrapped_sentence_highlighting_instance(instance).encode('utf-8'),
            '(%s:%d)' % (filename, sentence_num),
            file=log_file)

    @staticmethod
    def _print_with_labeled_args(instance, indent, out_file, arg_token_starts,
                                 arg_token_ends):
        '''
        Prints sentences annotated according to a particular instance.
        Connectives are printed in ALL CAPS. If run from a TTY, arguments are
        printed in color; otherwise, they're indicated as '/arg0/' and
        '*arg1*' (and _arg2_, if applicable).
        '''
        def get_printable_word(token):
            word = token.original_text
            if token in instance.connective:
                word = _get_printable_connective_word(word)
            for arg, token_start, token_end in zip(
                instance.get_args(), arg_token_starts, arg_token_ends):
                if arg and token in arg:
                    word = token_start + word + token_end
                    break
            return word

        tokens = instance.sentence.tokens[1:] # skip ROOT
        # TODO: should this be checking out_file?
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
                           sep='\n', file=out_file)

    def _log_arg_label_differences(self, indent, log_file):
        if sys.stdout.isatty() or FLAGS.iaa_force_color:
            arg_token_starts = [
                getattr(colorama.Fore, FLAGS.iaa_cause_color.upper()),
                getattr(colorama.Fore,FLAGS.iaa_effect_color.upper()),
                getattr(colorama.Fore, FLAGS.iaa_means_color.upper())]
            arg_token_ends = [colorama.Fore.RESET] * 3
        else:
            arg_token_starts = ['/', '*', '_']
            arg_token_ends = arg_token_starts

        for instance_1, instance_2, sentence_num in self.argument_differences:
            filename = os.path.split(instance_1.sentence.source_file_path)[-1]
            connective_text = StanfordParsedSentence.get_text_for_tokens(
                    instance_1.connective).encode('utf-8)')
            print_indented(
                indent, 'Arguments differ for connective "', connective_text,
                '" (', filename, ':', sentence_num, ')',
                ' with ', sep='', end='', file=log_file)
            arg_types = self._INSTANCE_CLASS.get_arg_types()
            for arg_type, arg_token_start, arg_token_end in zip(
                arg_types, arg_token_starts, arg_token_ends):
                print(arg_token_start, self._INSTANCE_CLASS.arg_names[arg_type],
                      arg_token_end, sep='', end='', file=log_file)
                if arg_type != arg_types[-1]: print(', ', end='', file=log_file)
            print(':', file=log_file)

            self._print_with_labeled_args(
                instance_1, indent + 1, log_file, arg_token_starts,
                arg_token_ends)
            # print_indented(indent + 1, "vs.", file=log_file)
            self._print_with_labeled_args(
                instance_2, indent + 1, log_file, arg_token_starts,
                arg_token_ends)

    def _log_property_differences(self, property_name, property_enum, indent,
                                  log_file):
        filtered_differences = [x for x in self.property_differences
                                if x[2] is property_enum]
        for instance_1, instance_2, _, sentence_num in filtered_differences:
            values = [property_enum[getattr(instance, property_name)]
                      for instance in [instance_1, instance_2]]
            filename = os.path.split(instance_1.sentence.source_file_path)[-1]
            encoded_instance = _wrapped_sentence_highlighting_instance(
                instance_1).encode('utf-8')
            print_indented(
                indent, property_name, 's for connective "',
                StanfordParsedSentence.get_text_for_tokens(
                    instance_1.connective).encode('utf-8)'),
                '" differ: ', values[0], ' vs. ', values[1], ' ',
                '(', filename, ':', sentence_num, ': "', encoded_instance, '")',
                sep='', file=log_file)

    @staticmethod
    def _csv_metrics(metrics_dict):
        lines = [',TP,FP,FN,S_c,H_c,J_c,S_e,H_e,J_e']
        for category, metrics in metrics_dict.iteritems():
            csv_metrics = (str(x) for x in [
                category,
                metrics.connective_metrics.tp,
                metrics.connective_metrics.fp,
                metrics.connective_metrics.fn,
                metrics.arg0_metrics.span_metrics.accuracy,
                metrics.arg0_metrics.head_metrics.accuracy,
                metrics.arg0_metrics.jaccard,
                metrics.arg1_metrics.span_metrics.accuracy,
                metrics.arg1_metrics.head_metrics.accuracy,
                metrics.arg1_metrics.jaccard])
            lines.append(','.join(csv_metrics))
        return '\n'.join(lines)


class CausalityMetrics(_RelationMetrics):
    _GOLD_INSTANCES_PROPERTY_NAME = 'causation_instances'
    _INSTANCE_CLASS = CausationInstance

    # TODO: Refactor order of parameters
    # TODO: provide both pairwise and non-pairwise stats
    def __init__(self, gold, predicted, allow_partial, save_differences=False,
                 ids_considered=None, compare_degrees=True, compare_types=True,
                 compare_args=True, pairwise_only=False, save_agreements=False,
                 compute_overlapping=None,
                 causations_property_name=_GOLD_INSTANCES_PROPERTY_NAME):
        properties_to_compare = [
            ('degree', CausationInstance.Degrees, compare_degrees),
            ('type', CausationInstance.CausationTypes, compare_types)]
        super(CausalityMetrics, self).__init__(
            gold, predicted, allow_partial, save_differences, ids_considered,
            compare_args, properties_to_compare, pairwise_only, save_agreements,
            causations_property_name)

        if compute_overlapping is None:
            compute_overlapping = FLAGS.iaa_compute_overlapping

        if compute_overlapping:
            self.overlapping = OverlappingRelMetrics(
                gold, predicted, allow_partial, save_differences,
                ids_considered, compare_types, compare_args, pairwise_only,
                save_agreements)
        else:
            self.overlapping = None

    def metrics_by_connective(self):
        by_connective = super(CausalityMetrics, self).metrics_by_connective()
        self._remap_by_connective(by_connective)
        return by_connective

    @staticmethod
    def _remap_by_connective(by_connective):
        to_remap = {'for too to':'too for to', 'for too':'too for',
                    'reason be':'reason', 'that now':'now that',
                    'to for':'for to', 'give':'given', 'thank to': 'thanks to',
                    'result of':'result', 'to need': 'need to'}
        for connective, metrics in by_connective.items():
            if connective.startswith('be '):
                by_connective[connective[3:]] += metrics
                del by_connective[connective]
                # print 'Replaced', connective
            elif connective in to_remap:
                by_connective[to_remap[connective]] += metrics
                del by_connective[connective]
                # print "Replaced", connective

    def metrics_by_connective_category(self):
        return self.get_aggregate_metrics(self.get_connective_category)

    def pp(self, log_confusion=None, log_stats=None, log_differences=None,
           log_agreements=None, log_by_connective=None, log_by_category=None,
           indent=0, log_file=sys.stdout):
        super(CausalityMetrics, self).pp(
            log_confusion, log_stats, log_differences, log_agreements,
            log_by_connective, indent, log_file)

        if log_by_category is None:
            log_by_category = FLAGS.iaa_log_by_category

        if log_by_category:
            print(file=log_file)
            print_indented(indent, 'Metrics by category:', file=log_file)
            by_category = self.metrics_by_connective_category()
            print_indented(indent + 1, self._csv_metrics(by_category),
                           file=log_file)

        if self.overlapping:
            print(file=log_file)
            print_indented(indent, 'Overlapping:', file=log_file)
            self.overlapping.pp(log_confusion, log_stats, log_differences,
                                log_agreements, log_by_connective, indent + 1,
                                log_file)

    __connective_types = merge_dicts([
        {'CC': 'Conjunctive (coordinating)', 'IN': 'Prepositional',
         'MD': 'Verbal', 'TO': 'Prepositional'},
        {'JJ' + suffix: 'Adjectival' for suffix in ['', 'R', 'S']},
        {'VB' + suffix: 'Verbal' for suffix in ['', 'D', 'G', 'N', 'P', 'Z']},
        {'RB' + suffix: 'Adverbial' for suffix in ['', 'R', 'S']},
        {'NN' + suffix: 'Nominal' for suffix in ['', 'S', 'P', 'PS']}])

    @staticmethod
    def get_connective_category(instance):
        connective = instance.connective

        # Treat if/thens like normal ifs
        if len(connective) == 1 or connective[1].lemma == 'then':
            connective = connective[0]
            if connective.pos == 'IN':
                edge_label, _parent = instance.sentence.get_most_direct_parent(
                    connective)
                if edge_label == 'mark' and connective.lemma != 'for':
                    return 'Conjunctive (subordinating)'
            return CausalityMetrics.__connective_types.get(connective.pos,
                                                           connective.pos)

        connective_head = instance.sentence.get_head(connective)
        # Special MWE cases: "because of", "thanks to", "now that", "out of"
        if len(connective) == 2:
            stringified = stringify_connective(instance)
            if stringified == 'because of':
                return 'Adverbial'
            elif stringified in ['thank to', 'now that']:
                return 'Conjunctive (subordinating)'
            elif stringified == 'out of':
                return 'Prepositional'

        # Anything tagged IN or TO is probably an argument realization word. If
        # there are non-argument-realization words in there, or if it's a
        # copula, it's complex.
        if any(t.lemma == 'be'
               or (t is not connective_head and (t.pos not in ['IN', 'TO']
                                                 or t.lemma in ['as', 'for']))
               for t in connective):
            return 'Complex'
        # A connective that's headed by a preposition or adverb and is otherwise
        # all prepositions is complex.
        elif connective_head.pos in ['IN', 'TO', 'RB', 'RBR', 'RBS']:
            return 'Complex'
        elif connective_head.pos.startswith('NN') and len(connective) > 2:
            return 'Complex'
        else:
            conn_type = CausalityMetrics.__connective_types[connective_head.pos]
            if (conn_type == 'Adjectival'
                and not StanfordParsedSentence.is_contiguous(connective)):
                return 'Complex'
            else:
                return conn_type

    def __add__(self, other):
        summed = super(CausalityMetrics, self).__add__(other)
        if self.overlapping and other.overlapping:
            summed.overlapping = self.overlapping + other.overlapping
        else:
            summed.overlapping = None
        return summed

    @staticmethod
    def aggregate(metrics_list):
        aggregated = _RelationMetrics.aggregate(metrics_list)
        all_overlapping = [m.overlapping for m in metrics_list]
        if None in all_overlapping:
            aggregated.overlapping = None
        else:
            aggregated.overlapping = reduce(operator.add, all_overlapping)
        return aggregated

# Map (cause|effect|means)_metrics to argi_metrics.
for underlying_name, arg_name in CausationInstance.arg_names.iteritems():
    underlying_name = underlying_name + '_metrics'
    getter = make_getter(underlying_name)
    setter = make_setter(underlying_name)
    setattr(CausalityMetrics, arg_name + '_metrics', property(getter, setter))


class OverlappingRelMetrics(_RelationMetrics):
    _INSTANCE_CLASS = OverlappingRelationInstance
    _GOLD_INSTANCES_PROPERTY_NAME = 'overlapping_rel_instances'

    def __init__(self, gold, predicted, allow_partial, save_differences=False,
                 ids_considered=None, compare_types=True, compare_args=True,
                 pairwise_only=False, save_agreements=False,
                 causations_property_name=_GOLD_INSTANCES_PROPERTY_NAME):
        properties_to_compare = [
            ('type', OverlappingRelationInstance.RelationTypes, compare_types)]
        super(OverlappingRelMetrics, self).__init__(
            gold, predicted, allow_partial, save_differences, ids_considered,
            compare_args, properties_to_compare, pairwise_only, save_agreements,
            causations_property_name)


def stringify_connective(instance):
    return ' '.join(t.lemma for t in instance.connective)
