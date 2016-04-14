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
    pipeline.train()
    eval_results = pipeline.evaluate()[0]
