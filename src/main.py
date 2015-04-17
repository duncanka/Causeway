#!/usr/bin/env python

import gflags
from sklearn import tree, neighbors, linear_model, svm, ensemble
import logging
import numpy as np
import os
import sys
from iaa import CausalityMetrics
from causality_pipelines.connective_based.crf_stage import ArgumentLabelerStage
FLAGS = gflags.FLAGS

from data.readers import DirectoryReader, StandoffReader
from pipeline import Pipeline
from pipeline.models import ClassBalancingModelWrapper
from causality_pipelines.pairwise.candidate_classifier import CandidateClassifierStage
from causality_pipelines.connective_based.regex_stage import RegexConnectiveStage
from causality_pipelines.pairwise.tregex_stage import TRegexConnectiveStage
from util.metrics import ClassificationMetrics

try:
    gflags.DEFINE_enum('pw_classifier_model', 'tree',
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
    gflags.DEFINE_enum('pipeline_type', 'tregex', ['tregex', 'regex'],
                       'Which causality pipeline to run')
except gflags.DuplicateFlagError as e:
    logging.warn('Ignoring redefinition of flag %s' % e.flagname)


# def main(argv):
if __name__ == '__main__':
    argv = sys.argv

    try:
        FLAGS(argv)  # parse flags
        # Print command line in case we ever want to re-run from output file
        print "Command line:", " ".join(
            [arg if ' ' not in arg else '"%s"' % arg
             for arg in argv[:]])
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

    if FLAGS.pipeline_type == 'tregex':
        if FLAGS.pw_classifier_model == 'tree':
            candidate_classifier = tree.DecisionTreeClassifier()
        elif FLAGS.pw_classifier_model == 'knn':
            candidate_classifier = neighbors.KNeighborsClassifier()
        elif FLAGS.pw_classifier_model == 'logistic':
            candidate_classifier = linear_model.LogisticRegression()
        elif FLAGS.pw_classifier_model == 'svm':
            candidate_classifier = svm.SVC()
        elif FLAGS.pw_classifier_model == 'forest':
            candidate_classifier = ensemble.RandomForestClassifier(n_jobs=-1)

        candidate_classifier = ClassBalancingModelWrapper(candidate_classifier,
                                                          FLAGS.rebalance_ratio)

        connective_stage = TRegexConnectiveStage('TRegex connectives')
        candidate_classifier_stage = CandidateClassifierStage(
            candidate_classifier, 'Candidate classifier')
        stages = [connective_stage, candidate_classifier_stage]
        results_names = ['All instances', 'Pairwise instances only']
        stage_aggregators = [TRegexConnectiveStage.average_eval_pairs,
                             ClassificationMetrics.average]
        # TODO: replace passing in aggregators with class-level variables
        # pointing to aggregator functions.
    else: # regex
        stages = [RegexConnectiveStage('Regex connectives'),
                  ArgumentLabelerStage('CRF arg labeler')]
        results_names = ['Allowing partial matches',
                         'Not allowing partial matches']
        stage_aggregators = [ClassificationMetrics.average,
                             lambda tups: (CausalityMetrics.aggregate(
                                               [tup[0] for tup in tups]),
                                           CausalityMetrics.aggregate(
                                               [tup[1] for tup in tups]))]

    causality_pipeline = Pipeline(
        stages, DirectoryReader((r'.*\.ann$',), StandoffReader()))

    def print_eval(eval_results):
        stage_names = [p.name for p in causality_pipeline.stages]
        for stage_name, result in zip(stage_names, eval_results):
            print "Evaluation for stage %s:" % stage_name
            # The labels will be used for eval results of the connective stage.
            causality_pipeline.print_stage_results(
                1, result, results_names)

    if FLAGS.eval_with_cv:
        logging.info("Evaluating with %d-fold cross-validation"
                     % FLAGS.cv_folds)
        eval_results = causality_pipeline.cross_validate(
            stage_aggregators=stage_aggregators)
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
