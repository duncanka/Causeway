import logging
import sys

from causality_pipelines.regex_based.regex_stage import RegexConnectiveStage
from causality_pipelines.tregex_based.tregex_stage import TRegexConnectiveStage
from gflags import FLAGS, FlagsError, DEFINE_enum, DuplicateFlagError
from data.io import DirectoryReader, CausalityStandoffReader
from pipeline import Pipeline

try:
    DEFINE_enum('pattern_type', 'tregex', ['tregex', 'regex'],
                'Which pattern type to test')
except DuplicateFlagError:
    pass

if __name__ == '__main__':
    try:
        FLAGS(sys.argv)
    except FlagsError, e:
        print '%s\\nUsage: %s ARGS\\n%s' % (e, sys.argv[0], FLAGS)
        sys.exit(1)

    logging.basicConfig(
        format='%(filename)s:%(lineno)s:%(levelname)s: %(message)s',
        level=logging.INFO)
    logging.captureWarnings(True)

    if FLAGS.pattern_type == 'tregex':
        stage = TRegexConnectiveStage('TRegex')
    else:
        stage = RegexConnectiveStage('Regex')

    pipeline = Pipeline(
        [stage],
        DirectoryReader((CausalityStandoffReader.FILE_PATTERN,),
                        CausalityStandoffReader()))

    FLAGS.test_paths = FLAGS.train_paths
    FLAGS.iaa_log_by_connective = True

    pipeline.train()
    eval_results = pipeline.evaluate()[0]

    by_connective = eval_results.metrics_by_connective()
    to_remap = {'for too to': 'too for to', 'for too': 'too for',
                'reason be': 'reason', 'that now': 'now that',
                'to for': 'for to', 'give': 'given', 'result of': 'result'}
    for connective, metrics in by_connective.items():
        if connective.startswith('be '):
            by_connective[connective[3:]] += metrics
            del by_connective[connective]
            print "Replaced", connective
        elif connective in to_remap:
            by_connective[to_remap[connective]] += metrics
            del by_connective[connective]
            print 'Replaced', connective

    for connective, metrics in sorted(by_connective.iteritems(),
                                      key=lambda pair: pair[0]):
        print ','.join([str(x) for x in connective, metrics.precision])
