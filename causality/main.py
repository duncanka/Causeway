#!/usr/bin/env python

import gflags
from sklearn import tree, neighbors, linear_model, svm
import sys
FLAGS = gflags.FLAGS

from data.readers import *
from pipeline import *
from pipeline.models import ClassBalancingModelWrapper
from simple_causality import SimpleCausalityStage
from util import metrics

try:
    gflags.DEFINE_enum('classifier_model', 'svm',
                       ['tree', 'knn', 'logistic', 'svm'],
                       'Which type of machine learning model to use as the'
                       ' underlying classifier')
    gflags.DEFINE_bool(
        'rebalance', True, 'Whether to rebalance classes for training')
except gflags.DuplicateFlagError as e:
    warnings.warn('Ignoring redefinition of flag %s' % e.flagname)


#def main(argv):
if __name__ == '__main__':
    argv = sys.argv
    try:
      FLAGS(argv)  # parse flags
    except gflags.FlagsError, e:
      print '%s\\nUsage: %s ARGS\\n%s' % (e, sys.argv[0], FLAGS)
      sys.exit(1)

    if FLAGS.classifier_model == 'tree':
        classifier = tree.DecisionTreeClassifier()
    elif FLAGS.classifier_model == 'knn':
        classifier = neighbors.KNeighborsClassifier()
    elif FLAGS.classifier_model == 'logistic':
        classifier = linear_model.LogisticRegression()
    elif FLAGS.classifier_model == 'svm':
        classifier = svm.SVC()

    if FLAGS.rebalance:
        classifier = ClassBalancingModelWrapper(classifier)

    causality_pipeline = Pipeline(
        SimpleCausalityStage(classifier),
        DirectoryReader((r'.*\.ann$',), StandoffReader()))

    if FLAGS.train_paths:
        causality_pipeline.train()

    if FLAGS.evaluate:
        eval_results = causality_pipeline.evaluate()
        stage_names = [p.name for p in causality_pipeline.stages]
        for stage_name, result in zip(stage_names, eval_results):
            print "Evaluation for stage %s:" % stage_name
            metrics.printer_indent_level += 1
            print result
            metrics.printer_indent_level -= 1
    elif FLAGS.test_paths:
        causality_pipeline.test()

#if __name__ == '__main__':
#    main(sys.argv)
