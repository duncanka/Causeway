#!/usr/bin/env python

import gflags
import logging
import numpy as np
import os
from sklearn import tree, neighbors, linear_model, svm, ensemble
import sys

from causality_pipelines.baseline import BaselineStage
from causality_pipelines.baseline.combiner import BaselineCombinerStage
from causality_pipelines.regex_based.candidate_classifier import RegexClassifierStage
from causality_pipelines.regex_based.crf_stage import ArgumentLabelerStage
from causality_pipelines.regex_based.regex_stage import RegexConnectiveStage
from causality_pipelines.tregex_based.candidate_classifier import TRegexClassifierStage
from causality_pipelines.tregex_based.tregex_stage import TRegexConnectiveStage
from data import ParsedSentence
from data.readers import DirectoryReader, StandoffReader
from pipeline import Pipeline
from pipeline.models import ClassBalancingModelWrapper
from util import print_indented

FLAGS = gflags.FLAGS


try:
    gflags.DEFINE_enum('classifier_model', 'forest',
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
    gflags.DEFINE_enum('pipeline_type', 'tregex',
                       ['tregex', 'regex', 'baseline', 'baseline+tregex',
                        'baseline+regex'],
                       'Which causality pipeline to run')
except gflags.DuplicateFlagError as e:
    logging.warn('Ignoring redefinition of flag %s' % e.flagname)


def get_stages(candidate_classifier):
    BASELINE_CAUSATIONS_NAME = 'baseline_causations'

    if FLAGS.pipeline_type == 'tregex':
        stages = [TRegexConnectiveStage('TRegex connectives'),
                  TRegexClassifierStage(candidate_classifier,
                                        'Candidate classifier')]
    elif FLAGS.pipeline_type == 'regex':
        stages = [RegexConnectiveStage('Regex connectives'),
                  ArgumentLabelerStage('CRF arg labeler'),
                  RegexClassifierStage(candidate_classifier,
                                       'Candidate classifier')]
    elif FLAGS.pipeline_type == 'baseline+tregex':
        stages = [BaselineStage('Baseline', BASELINE_CAUSATIONS_NAME),
                  TRegexConnectiveStage('TRegex connectives'),
                  TRegexClassifierStage(candidate_classifier,
                                        'Candidate classifier'),
                  BaselineCombinerStage('Combiner', BASELINE_CAUSATIONS_NAME)]
    elif FLAGS.pipeline_type == 'baseline+regex':
        stages = [BaselineStage('Baseline', BASELINE_CAUSATIONS_NAME),
                  RegexConnectiveStage('Regex connectives'),
                  ArgumentLabelerStage('CRF arg labeler'),
                  RegexClassifierStage(candidate_classifier,
                                       'Candidate classifier'),
                  BaselineCombinerStage('Combiner', BASELINE_CAUSATIONS_NAME)]
    else:
        stages = [BaselineStage('Baseline')] # baseline
    return stages

# def main(argv):
if __name__ == '__main__':
    argv = sys.argv

    try:
        FLAGS(argv)  # parse flags
        # Print command line in case we ever want to re-run from output file
        print "Flags:"
        print_indented(1, FLAGS.FlagsIntoString())
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

    if FLAGS.classifier_model == 'tree':
        candidate_classifier = tree.DecisionTreeClassifier()
    elif FLAGS.classifier_model == 'knn':
        candidate_classifier = neighbors.KNeighborsClassifier()
    elif FLAGS.classifier_model == 'logistic':
        candidate_classifier = linear_model.LogisticRegression(penalty='l1')
    elif FLAGS.classifier_model == 'svm':
        candidate_classifier = svm.SVC()
    elif FLAGS.classifier_model == 'forest':
        candidate_classifier = ensemble.RandomForestClassifier(n_jobs=-1)
    
    candidate_classifier = ClassBalancingModelWrapper(candidate_classifier,
                                                      FLAGS.rebalance_ratio)

    stages = get_stages(candidate_classifier)

    causality_pipeline = Pipeline(
        stages, DirectoryReader((r'.*\.ann$',), StandoffReader()),
        copy_fn=ParsedSentence.shallow_copy_with_sentences_fixed)

    if FLAGS.eval_with_cv:
        eval_results = causality_pipeline.cross_validate()
        causality_pipeline.print_eval_results(eval_results)
    else:
        if FLAGS.train_paths:
            causality_pipeline.train()

        if FLAGS.evaluate:
            eval_results = causality_pipeline.evaluate()
            causality_pipeline.print_eval_results(eval_results)
        elif FLAGS.test_paths:
            causality_pipeline.test()

#if __name__ == '__main__':
#    main(sys.argv)
