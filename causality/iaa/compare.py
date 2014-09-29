from __future__ import print_function
from gflags import *
import logging
import sys

from data.readers import *
from iaa import *
from util import *

try:
    DEFINE_list(
        'iaa_paths', [], "Paths to annotation files to be compared against each"
        " other for IAA.")
    DEFINE_list(
        'iaa_file_regexes', r".*\.ann",
        "Regexes to match filenames against for IAA (non-matching files will"
        " not be compared).")
    DEFINE_integer('iaa_max_sentence', sys.maxint,
                   'Maximum number of sentences to analyze when computing IAA.')
except DuplicateFlagError as e:
    logging.warn('Ignoring redefinition of flag %s' % e.flagname)


def compare_instance_lists(gold, predicted, indent=0):
    printing_some_metrics = FLAGS.iaa_log_confusion or FLAGS.iaa_log_stats

    for allow_partial in [True, False]:
        non_given_only_metrics = CausalityMetrics(
            gold, predicted, allow_partial, False,
            CausalityMetrics.IDsConsidered.NonGivenOnly)

        print_indented(indent, ('%sllowing partial matches:'
                                % ['Not a', 'A'][allow_partial]))

        if printing_some_metrics:
            indent += 1
            if FLAGS.iaa_given_connective_ids:
                print_indented(indent, 'Without gold connectives:')
            non_given_only_metrics.pp(
                FLAGS.iaa_log_confusion, FLAGS.iaa_log_stats,
                False, indent + 1)

        all_metrics = CausalityMetrics(
            gold, predicted, allow_partial, FLAGS.iaa_log_differences,
            CausalityMetrics.IDsConsidered.Both)

        if FLAGS.iaa_given_connective_ids:
            given_only_metrics = CausalityMetrics(
                gold, predicted, allow_partial, False,
                CausalityMetrics.IDsConsidered.GivenOnly)
            if printing_some_metrics:
                print()
                print_indented(indent, 'With only gold connectives:')
            given_only_metrics.pp(
                FLAGS.iaa_log_confusion, FLAGS.iaa_log_stats,
                False, indent + 1)

            if printing_some_metrics:
                print()
                print_indented(indent, 'Counting all connectives:')

        all_metrics.pp(
            FLAGS.iaa_log_confusion, FLAGS.iaa_log_stats,
            FLAGS.iaa_log_differences, indent + 1)

        if printing_some_metrics:
            indent -= 1


def main(argv):
    try:
        FLAGS(argv)  # parse flags
    except FlagsError as e:
        print('%s\nUsage: %s ARGS\n%s' % (e, sys.argv[0], FLAGS))
        sys.exit(1)

    if not (FLAGS.iaa_log_differences or FLAGS.iaa_log_stats
        or FLAGS.iaa_log_confusion):
        print('Nothing to log; exiting')
        sys.exit(0)

    reader = DirectoryReader(FLAGS.iaa_file_regexes, StandoffReader())
    all_instances = []
    for path in FLAGS.iaa_paths:
        reader.open(path)
        all_instances.append(reader.get_all()[:FLAGS.iaa_max_sentence])

    instances_path_pairs = zip(all_instances, FLAGS.iaa_paths)
    for ((gold, gold_path), (predicted, predicted_path)) in (
            itertools.combinations(instances_path_pairs, 2)):
        print('%s vs. %s:' % (gold_path, predicted_path))
        compare_instance_lists(gold, predicted, 1)
        print()

if __name__ == '__main__':
    main(sys.argv)
