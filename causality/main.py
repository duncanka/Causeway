#!/usr/bin/env python

import gflags
import sys
FLAGS = gflags.FLAGS

from data.readers import *
from pipeline import *
from simple_causality import SimpleCausalityStage
from util import metrics

def main(argv):
    try:
      FLAGS(argv)  # parse flags
    except gflags.FlagsError, e:
      print '%s\\nUsage: %s ARGS\\n%s' % (e, sys.argv[0], FLAGS)
      sys.exit(1)

    causality_pipeline = Pipeline(
        SimpleCausalityStage(),
        DirectoryReader((r'.*\.ann$',), StandoffReader()))

    if FLAGS.train_paths:
        causality_pipeline.train()

    if FLAGS.evaluate:
        # TODO: sort more intelligently?
        eval_results = causality_pipeline.evaluate()
        stage_names = [p.name for p in causality_pipeline.stages]
        for stage_name, result in zip(stage_names, eval_results):
            print "Evaluation for stage %s:" % stage_name
            metrics.printer_indent_level += 1
            print result
            metrics.printer_indent_level -= 1
    elif FLAGS.test_paths:
        causality_pipeline.test()

if __name__ == '__main__':
    main(sys.argv)
