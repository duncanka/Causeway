#!/usr/bin/env python

from __future__ import print_function

from gflags import (DEFINE_list, DEFINE_bool, DEFINE_integer,
                    DuplicateFlagError, FlagsError, FLAGS)
import itertools
import logging
import sys

from causeway.because_data import CausalityStandoffReader
from data.io import DirectoryReader
from iaa import CausalityMetrics, print_indented


try:
    DEFINE_list(
        'iaa_file_regexes', r".*\.ann$",
        "Regexes to match filenames against for IAA (non-matching files will"
        " not be compared).")
    DEFINE_integer('iaa_max_sentence', sys.maxint,
                   'Maximum number of sentences to analyze when computing IAA.')
    DEFINE_bool('iaa_include_partial', False,
                'Include a comparison that counts partial overlap of spans as a'
                ' match.')
except DuplicateFlagError as e:
    logging.warn('Ignoring flag redefinitions; assuming module reload')


def compare_instance_lists(gold, predicted, indent=0):
    printing_some_metrics = FLAGS.iaa_log_confusion or FLAGS.iaa_log_stats

    if FLAGS.iaa_include_partial:
        partial_possibilities = [True, False]
    else:
        partial_possibilities = [False]

    for allow_partial in partial_possibilities:
        print_indented(indent, ('%sllowing partial matches:'
                                % ['Not a', 'A'][allow_partial]))

        # Only do given/non-given sections if there are some given IDs.
        if FLAGS.iaa_given_connective_ids and printing_some_metrics:
            non_given_only_metrics = CausalityMetrics(
                gold, predicted, allow_partial, False,
                CausalityMetrics.IDsConsidered.NonGivenOnly)
            indent += 1
            print_indented(indent, 'Without gold connectives:')
            non_given_only_metrics.pp(log_differences=False,
                                      log_agreements=False, indent=indent + 1)

            given_only_metrics = CausalityMetrics(
                gold, predicted, allow_partial, False,
                CausalityMetrics.IDsConsidered.GivenOnly)
            print_indented(indent, 'With only gold connectives:')
            given_only_metrics.pp(log_differences=False,
                                  log_agreements=False, indent=indent + 1)

            print()
            print_indented(indent, 'Counting all connectives:')


        all_metrics = CausalityMetrics(
            gold, predicted, allow_partial, FLAGS.iaa_log_differences,
            CausalityMetrics.IDsConsidered.Both)
        all_metrics.pp(log_agreements=False, indent=indent + 1)

        # Restore indent only if we increased it earlier.
        if FLAGS.iaa_given_connective_ids and printing_some_metrics:
            indent -= 1

        print()


def main(argv):
    try:
        argv = FLAGS(argv) # parse flags
        iaa_paths = argv[1:]
    except FlagsError as e:
        print('%s\nUsage: %s ARGS\n%s' % (e, sys.argv[0], FLAGS))
        sys.exit(1)

    if not (FLAGS.iaa_log_differences or FLAGS.iaa_log_stats
            or FLAGS.iaa_log_confusion):
        print('Nothing to log for comparison')
        sys.exit(0)

    logging.basicConfig(
        format='%(filename)s:%(lineno)s:%(levelname)s: %(message)s',
        level=logging.WARN)
    logging.captureWarnings(True)

    reader = DirectoryReader(FLAGS.iaa_file_regexes, CausalityStandoffReader())
    instances_by_path = []
    for path in iaa_paths:
        reader.open(path)
        path_docs = reader.get_all()[:FLAGS.iaa_max_sentence]
        path_instances = list(itertools.chain.from_iterable(doc.sentences for
                                                            doc in path_docs))
        instances_by_path.append(path_instances)
    reader.close()

    instances_path_pairs = zip(instances_by_path, iaa_paths)
    for ((gold, gold_path), (predicted, predicted_path)) in (
            itertools.combinations(instances_path_pairs, 2)):
        print('%s vs. %s:' % (gold_path, predicted_path))
        compare_instance_lists(gold, predicted, 1)
        print()

if __name__ == '__main__':
    main(sys.argv)
