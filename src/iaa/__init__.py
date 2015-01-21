from __future__ import print_function
from collections import defaultdict
from gflags import *
import logging
import os
import sys

from data import *
from util import *
from util.diff import SequenceDiff
from util.metrics import *

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


def get_truncated_sentence(instance):
    return truncated_string(
        instance.source_sentence.original_text.replace('\n', ' '))

class CausalityMetrics(object):
    IDsConsidered = Enum(['GivenOnly', 'NonGivenOnly', 'Both'])
    ArgTypes = Enum(['Cause', 'Effect'])

    def __init__(self, gold, predicted, allow_partial,
                 save_differences=False, ids_considered=None):
        if ids_considered is None:
            ids_considered = CausalityMetrics.IDsConsidered.Both
        self.allow_partial = allow_partial
        self._annotation_comparator = make_annotation_comparator(allow_partial)
        self.ids_considered = ids_considered
        self.save_differences = save_differences
        self.gold_only_instances = []
        self.predicted_only_instances = []
        self.property_differences = []

        assert len(gold) == len(predicted), (
            "Cannot compute IAA for different-sized datasets")

        # Compute attributes that take a little more work.
        self.connective_metrics, matches = self._match_connectives(
            gold, predicted)
        self.degree_matrix = self._compute_agreement_matrix(
            matches, CausationInstance.Degrees, 'degree', gold)
        self.causation_type_matrix = self._compute_agreement_matrix(
            matches, CausationInstance.CausationTypes, 'type', gold)
        self.arg_metrics, self.arg_label_matrix = self._match_arguments(
            matches, gold)

    def __get_causations(self, sentence):
        causations = []
        for instance in sentence.causation_instances:
            is_given_id = instance.id in FLAGS.iaa_given_connective_ids
            if (self.ids_considered == self.IDsConsidered.Both or
                (is_given_id and
                 self.ids_considered == self.IDsConsidered.GivenOnly) or
                (not is_given_id and
                 self.ids_considered == self.IDsConsidered.NonGivenOnly)):
                causations.append(instance)
        sort_key = lambda inst: inst.connective[0].start_offset
        return sorted(causations, key=sort_key)

    def _match_connectives(self, gold, predicted):
        matching_instances = []
        gold_only_instances = []
        predicted_only_instances = []
        def compare_connectives(instance_1, instance_2):
            return self._annotation_comparator(instance_1.connective,
                                               instance_2.connective)

        for gold_sentence, predicted_sentence in zip(gold, predicted):
            assert (gold_sentence.original_text ==
                    predicted_sentence.original_text), (
                        "Can't compare annotations on non-identical sentences")
            gold_causations = self.__get_causations(gold_sentence)
            predicted_causations = self.__get_causations(predicted_sentence)
            diff = SequenceDiff(gold_causations, predicted_causations,
                                compare_connectives)
            matching_instances.extend(diff.get_matching_pairs())
            gold_only_instances.extend(diff.get_a_only_elements())
            predicted_only_instances.extend(diff.get_b_only_elements())

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

        if self.save_differences:
            def sentences_by_file(sentences):
                by_file = defaultdict(list)
                for sentence in sentences:
                    filename = os.path.split(sentence.source_file_path)[-1]
                    by_file[filename].append(sentence)
                return by_file
            gold_by_file = sentences_by_file(gold)
            predicted_by_file = sentences_by_file(predicted)

            self.gold_only_instances = [
                (gold_by_file[os.path.split(
                    i.source_sentence.source_file_path)[-1]]
                 .index(i.source_sentence) + 1, i)
                for i in gold_only_instances]
            self.predicted_only_instances = [
                (predicted_by_file[os.path.split(
                    i.source_sentence.source_file_path)[-1]]
                 .index(i.source_sentence) + 1, i)
                for i in predicted_only_instances]

        return (connective_metrics, matching_instances)

    def _compute_agreement_matrix(self, matches, labels_enum, property_name,
                                  gold_sentences):
        labels_1 = []
        labels_2 = []

        def log_missing(instance, number):
            print(property_type_name,
                  ('property not set in Annotation %d;' % number),
                  'not including in analysis (sentence: "',
                  get_truncated_sentence(instance_1) + '")',
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
                sentence_num = (
                    gold_sentences.index(instance_1.source_sentence) + 1)
                if property_1 != property_2 and self.save_differences:
                    self.property_differences.append(
                        (instance_1, instance_2, labels_enum, sentence_num))

        return ConfusionMatrix(labels_1, labels_2)

    def _match_arguments(self, matches, gold):
        # Initially, we assume every argument was unique. We'll update this
        # incrementally as we find matches.
        gold_only_args = predicted_only_args = 2 * len(matches)
        null_args = 0

        gold_labels = []
        predicted_labels = []

        for instance_1, instance_2 in matches:
            gold_args = (instance_1.cause, instance_1.effect)
            predicted_args = (instance_2.cause, instance_2.effect)
            sentence_num = gold.index(instance_1.source_sentence) + 1

            predicted_args_matched = [False, False]
            for i in range(len(gold_args)):
                if gold_args[i] is None:
                    gold_only_args -= 1
                    null_args += 1
                    continue
                for j in range(len(predicted_args)):
                    if predicted_args[j] is None:
                        # Only update arg counts on the first round to avoid
                        # double-counting
                        if i == 0:
                            predicted_only_args -= 1
                            null_args += 1
                        continue
                    elif predicted_args_matched[j]:
                        continue
                    elif self._annotation_comparator(gold_args[i],
                                                     predicted_args[j]):
                        gold_labels.append(self.ArgTypes[i])
                        predicted_labels.append(self.ArgTypes[j])
                        if self.save_differences and i != j:
                            self.property_differences.append(
                                (instance_1, instance_2, self.ArgTypes,
                                 sentence_num))
                        predicted_args_matched[j] = True
                        gold_only_args -= 1
                        predicted_only_args -= 1
                        # We're done matching this gold arg; move on to the next
                        break


        total_matches = len(gold_labels)
        #assert 4 * len(matches) == (2 * total_matches + gold_only_args +
        #                            predicted_only_args + null_args)
        arg_metrics = ClassificationMetrics(total_matches, predicted_only_args,
                                            gold_only_args)
        arg_label_matrix = ConfusionMatrix(gold_labels, predicted_labels)
        return arg_metrics, arg_label_matrix

    def pp(self, log_confusion=None, log_stats=None, log_differences=None,
           indent=0):
        # Flags aren't available as defaults when the function is created, so
        # set the defaults here.
        if log_confusion is None:
            log_confusion = FLAGS.iaa_log_confusion
        if log_stats is None:
            log_stats = FLAGS.iaa_log_stats
        if log_differences is None:
            log_differences = FLAGS.iaa_log_differences

        if log_differences:
            print_indented(indent, 'Annotation differences:')
            for sentence_num, instance in self.gold_only_instances:
                self._log_unique_instance(instance, sentence_num, 1,
                                          indent + 1)
            for sentence_num, instance in self.predicted_only_instances:
                self._log_unique_instance(instance, sentence_num, 2,
                                           indent + 1)
            self._log_property_differences(CausationInstance.CausationTypes,
                                           indent + 1)
            self._log_property_differences(CausationInstance.Degrees,
                                           indent + 1)
            self._log_property_differences(self.ArgTypes, indent + 1)

        # Ignore connective-related metrics if we have nothing interesting to
        # show there.
        printing_connective_metrics = (log_stats and self.connective_metrics)
        if printing_connective_metrics or log_confusion:
            print_indented(indent, 'Connectives:')
        if printing_connective_metrics:
            print_indented(indent + 1, self.connective_metrics)
        if log_stats or log_confusion:
            self._log_property_metrics('Degrees', self.degree_matrix,
                                       indent + 1, log_confusion, log_stats)
            self._log_property_metrics(
                'Causation types', self.causation_type_matrix, indent + 1,
                log_confusion, log_stats)

        if log_stats:
            print_indented(indent, 'Arguments:')
            print_indented(indent + 1, self.arg_metrics)

        if log_stats or log_confusion:
            self._log_property_metrics('Argument labels', self.arg_label_matrix,
                                       indent + 1, log_confusion, log_stats)

    def _log_property_metrics(self, name, matrix, indent, log_confusion,
                              log_stats):
        print_indented(indent, name, ':', sep='')
        if log_confusion:
            print_indented(indent + 1, matrix.pp(metrics=log_stats))
        else: # we must be logging just stats
            print_indented(indent + 1, matrix.pp_metrics())


    @staticmethod
    def _log_unique_instance(instance, sentence_num, annotator_num, indent):
        connective_text = ParsedSentence.get_annotation_text(
            instance.connective)
        filename = os.path.split(instance.source_sentence.source_file_path)[-1]
        print_indented(
            indent, "Annotation", annotator_num,
            'only: "%s"' % connective_text, '(%s:%d: "%s")'
            % (filename, sentence_num, get_truncated_sentence(instance)))

    def _log_property_differences(self, property_enum, indent):
        if property_enum is self.ArgTypes:
            property_name = 'Argument label'
            value_extractor = None
        elif property_enum is CausationInstance.Degrees:
            property_name = 'Degree'
            value_extractor = lambda instance: instance.Degrees[instance.degree]
        elif property_enum is CausationInstance.CausationTypes:
            property_name = 'Causation type'
            value_extractor = lambda instance: (
                instance.CausationTypes[instance.type])

        filtered_differences = [x for x in self.property_differences
                                if x[2] is property_enum]
        for instance_1, instance_2, _, sentence_num in filtered_differences:
            if value_extractor:
                values = (value_extractor(instance_1),
                          value_extractor(instance_2))
            filename = os.path.split(
                instance_1.source_sentence.source_file_path)[-1]
            print_indented(
                indent, property_name, 's for connective "',
                ParsedSentence.get_annotation_text(instance_1.connective),
                '" differ',
                (': %s vs. %s' % values if value_extractor else ''),
                ' (', filename, ':', sentence_num, ': "',
                get_truncated_sentence(instance_1), '")', sep='')
