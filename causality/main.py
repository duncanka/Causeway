#!/usr/bin/env python

import gflags
from sklearn import tree, neighbors, linear_model, svm, ensemble
import logging
import numpy as np
import os
import sys
FLAGS = gflags.FLAGS

from data.readers import DirectoryReader, StandoffReader
from pipeline import Pipeline
from pipeline.models import ClassBalancingModelWrapper
from pairwise.simple_causality import SimpleCausalityStage
from pairwise.connective_stage import ConnectiveStage
from util import print_indented
from util.metrics import ClassificationMetrics

try:
    gflags.DEFINE_enum('sc_classifier_model', 'forest',
                       ['tree', 'knn', 'logistic', 'svm', 'forest'],
                       'What type of machine learning model to use as the'
                       ' underlying simple causality classifier')
    gflags.DEFINE_float(
        'rebalance_ratio', 1.0,
        'The maximum ratio by which to rebalance classes for training')
    gflags.DEFINE_bool('eval_with_cv', False,
                       'Evaluate with cross-validation. Overrides --evaluate'
                       ' flag, and causes both train and test to be combined.')
    gflags.DEFINE_bool('debug', False,
                       'Whether to print debug-level logging.')
    gflags.DEFINE_integer('seed', None, 'Seed for the numpy RNG.')
except gflags.DuplicateFlagError as e:
    logging.warn('Ignoring redefinition of flag %s' % e.flagname)


# def main(argv):
if __name__ == '__main__':
    argv = sys.argv
    try:
        FLAGS(argv)  # parse flags
    except gflags.FlagsError, e:
        print '%s\\nUsage: %s ARGS\\n%s' % (e, sys.argv[0], FLAGS)
        sys.exit(1)

    logging.basicConfig(
        format='%(filename)s:%(lineno)s:%(levelname)s: %(message)s',
        level=[logging.INFO, logging.DEBUG][FLAGS.debug])
    logging.captureWarnings(True)

    seed = FLAGS.seed
    if seed is None:
        seed = int(os.urandom(4).encode('hex'), 16)
    np.random.seed(seed)
    print "Using seed:", seed

    if FLAGS.sc_classifier_model == 'tree':
        sc_classifier = tree.DecisionTreeClassifier()
    elif FLAGS.sc_classifier_model == 'knn':
        sc_classifier = neighbors.KNeighborsClassifier()
    elif FLAGS.sc_classifier_model == 'logistic':
        sc_classifier = linear_model.LogisticRegression()
    elif FLAGS.sc_classifier_model == 'svm':
        sc_classifier = svm.SVC()
    elif FLAGS.sc_classifier_model == 'forest':
        sc_classifier = ensemble.RandomForestClassifier(n_jobs=-1)

    sc_classifier = ClassBalancingModelWrapper(sc_classifier,
                                               FLAGS.rebalance_ratio)

    connective_stage = ConnectiveStage('Connectives')
    sc_stage = SimpleCausalityStage(sc_classifier)
    causality_pipeline = Pipeline(
        [connective_stage, sc_stage],
        DirectoryReader((r'.*\.ann$',), StandoffReader()))

    def print_eval(eval_results):
        stage_names = [p.name for p in causality_pipeline.stages]
        for stage_name, result in zip(stage_names, eval_results):
            print "Evaluation for stage %s:" % stage_name
            # The labels will be used for eval results of the connective stage.
            causality_pipeline.print_stage_results(
                1, result, ['All instances', 'Pairwise instances only'])

    if FLAGS.eval_with_cv:
        print "Evaluating with %d-fold cross-validation" % FLAGS.cv_folds
        eval_results = causality_pipeline.cross_validate(
            stage_aggregators=[ConnectiveStage.average_eval_pairs,
                               ClassificationMetrics.average])
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
