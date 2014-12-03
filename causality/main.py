#!/usr/bin/env python

import gflags
from sklearn import tree, neighbors, linear_model, svm, ensemble
import logging
import sys
FLAGS = gflags.FLAGS

from data.readers import DirectoryReader, StandoffReader
from pipeline import Pipeline
from pipeline.models import ClassBalancingModelWrapper
from simple_causality import SimpleCausalityStage
from util import print_indented
from util.metrics import ClassificationMetrics

try:
    gflags.DEFINE_enum('classifier_model', 'svm',
                       ['tree', 'knn', 'logistic', 'svm', 'forest'],
                       'Which type of machine learning model to use as the'
                       ' underlying classifier')
    gflags.DEFINE_float(
        'rebalance_ratio', 1.0,
        'The maximum ratio by which to rebalance classes for training')
    gflags.DEFINE_bool('eval_with_cv', False,
                       'Evaluate with cross-validation. Overrides --evaluate'
                       ' flag, and causes both train and test to be combined.')
except gflags.DuplicateFlagError as e:
    logging.warn('Ignoring redefinition of flag %s' % e.flagname)


#def main(argv):
if __name__ == '__main__':
    argv = sys.argv
    try:
        FLAGS(argv)  # parse flags
    except gflags.FlagsError, e:
        print '%s\\nUsage: %s ARGS\\n%s' % (e, sys.argv[0], FLAGS)
        sys.exit(1)

    logging.basicConfig(
        format='%(filename)s:%(lineno)s:%(levelname)s: %(message)s',
        level=logging.DEBUG)
    logging.captureWarnings(True)

    if FLAGS.classifier_model == 'tree':
        classifier = tree.DecisionTreeClassifier()
    elif FLAGS.classifier_model == 'knn':
        classifier = neighbors.KNeighborsClassifier()
    elif FLAGS.classifier_model == 'logistic':
        classifier = linear_model.LogisticRegression()
    elif FLAGS.classifier_model == 'svm':
        classifier = svm.SVC()
    elif FLAGS.classifier_model == 'forest':
        classifier = ensemble.RandomForestClassifier(n_jobs=-1)

    classifier = ClassBalancingModelWrapper(classifier, FLAGS.rebalance_ratio)

    causality_pipeline = Pipeline(
        SimpleCausalityStage(classifier),
        DirectoryReader((r'.*\.ann$',), StandoffReader()))

    def print_eval(eval_results):
        stage_names = [p.name for p in causality_pipeline.stages]
        for stage_name, result in zip(stage_names, eval_results):
            print "Evaluation for stage %s:" % stage_name
            print_indented(1, result)

    if FLAGS.eval_with_cv:
        print "Evaluating with %d-fold cross-validation" % FLAGS.cv_folds
        eval_results = causality_pipeline.cross_validate(
            stage_aggregators=[ClassificationMetrics.average])
        print_eval(eval_results)
    else:
        if FLAGS.train_paths:
            causality_pipeline.train()

        if FLAGS.evaluate:
            eval_results = causality_pipeline.evaluate()
            print_eval(eval_results)
        elif FLAGS.test_paths:
            causality_pipeline.test()

#if __name__ == '__main__':
#    main(sys.argv)
