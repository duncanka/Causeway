#!/usr/bin/env python

import gflags
import logging
import numpy as np
import os
from sklearn import tree, neighbors, linear_model, svm, ensemble, naive_bayes
import sys

from causality_pipelines.baseline import BaselineStage
from causality_pipelines.baseline.combiner import BaselineCombinerStage
from causality_pipelines.candidate_filter import CausationPatternFilterStage
from causality_pipelines.regex_based.crf_stage import ArgumentLabelerStage
from causality_pipelines.regex_based.regex_stage import RegexConnectiveStage
from causality_pipelines.tregex_based.arg_span_stage import ArgSpanStage
from causality_pipelines.tregex_based.tregex_stage import TRegexConnectiveStage
from data import StanfordParsedSentence
from data.io import DirectoryReader, CausalityStandoffReader
from pipeline import Pipeline
from pipeline.models import ClassBalancingClassifierWrapper
from util import print_indented
import subprocess
from causality_pipelines.baseline.most_freq_filter import MostFreqSenseFilterStage

FLAGS = gflags.FLAGS


try:
    gflags.DEFINE_enum('classifier_model', 'forest',
                       ['tree', 'knn', 'logistic', 'svm', 'forest', 'nb'],
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
                        'baseline+regex', 'tregex_mostfreq', 'regex_mostfreq'],
                       'Which causality pipeline to run')
except gflags.DuplicateFlagError as e:
    logging.warn('Ignoring redefinition of flag %s' % e.flagname)


def get_stages(candidate_classifier):
    BASELINE_CAUSATIONS_NAME = 'baseline_causations'

    if FLAGS.pipeline_type == 'tregex':
        stages = [TRegexConnectiveStage('TRegex connectives'),
                  ArgSpanStage('Argument span expander'),
                  CausationPatternFilterStage(candidate_classifier,
                                              'Candidate classifier')]
    elif FLAGS.pipeline_type == 'regex':
        stages = [RegexConnectiveStage('Regex connectives'),
                  ArgumentLabelerStage('CRF arg labeler'),
                  CausationPatternFilterStage(candidate_classifier,
                                              'Candidate classifier')]
    elif FLAGS.pipeline_type == 'baseline+tregex':
        stages = [BaselineStage('Baseline', BASELINE_CAUSATIONS_NAME),
                  TRegexConnectiveStage('TRegex connectives'),
                  ArgSpanStage('Argument span expander'),
                  CausationPatternFilterStage(candidate_classifier,
                                              'Candidate classifier'),
                  BaselineCombinerStage('Combiner', BASELINE_CAUSATIONS_NAME)]
    elif FLAGS.pipeline_type == 'baseline+regex':
        stages = [BaselineStage('Baseline', BASELINE_CAUSATIONS_NAME),
                  RegexConnectiveStage('Regex connectives'),
                  ArgumentLabelerStage('CRF arg labeler'),
                  CausationPatternFilterStage(candidate_classifier,
                                              'Candidate classifier'),
                  BaselineCombinerStage('Combiner', BASELINE_CAUSATIONS_NAME)]
    elif FLAGS.pipeline_type == 'baseline':
        stages = [BaselineStage('Baseline')]
    elif FLAGS.pipeline_type == 'tregex_mostfreq':
        stages = [TRegexConnectiveStage('TRegex'),
                  MostFreqSenseFilterStage('Most frequent sense filter')]
    elif FLAGS.pipeline_type == 'regex_mostfreq':
        stages = [RegexConnectiveStage('Regex'),
                  MostFreqSenseFilterStage('Most frequent sense filter')]
    return stages


def remap_by_connective(by_connective):
    to_remap = {'for too to':'too for to', 'for too':'too for',
                'reason be':'reason', 'that now':'now that', 'to for':'for to',
                'give':'given', 'thank to': 'thanks to', 'result of':'result',
                'to need': 'need to'}
    for connective, metrics in by_connective.items():
        if connective.startswith('be '):
            by_connective[connective[3:]] += metrics
            del by_connective[connective]
            # print 'Replaced', connective
        elif connective in to_remap:
            by_connective[to_remap[connective]] += metrics
            del by_connective[connective]
            # print "Replaced", connective


# def main(argv):
if __name__ == '__main__':
    argv = sys.argv

    try:
        FLAGS(argv)  # parse flags
        # Print command line in case we ever want to re-run from output file
        print "Flags:"
        print_indented(1, FLAGS.FlagsIntoString())
        print "Git info:"
        print_indented(1, subprocess.check_output("git rev-parse HEAD".split()),
                       "Modified:", sep='')
        print_indented(2, subprocess.check_output("git ls-files -m".split()))
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
    elif FLAGS.classifier_model == 'nb':
        candidate_classifier = naive_bayes.MultinomialNB()

    candidate_classifier = ClassBalancingClassifierWrapper(
        candidate_classifier, FLAGS.rebalance_ratio)

    stages = get_stages(candidate_classifier)

    causality_pipeline = Pipeline(
        stages, DirectoryReader((CausalityStandoffReader.FILE_PATTERN,),
                                CausalityStandoffReader()),
        copy_fn=StanfordParsedSentence.shallow_copy_doc_with_sentences_fixed)

    if FLAGS.eval_with_cv:
        eval_results = causality_pipeline.cross_validate()
        if FLAGS.log_connective_stats:
            print
            for stage, eval_metrics in zip(causality_pipeline.stages[1:],
                                           eval_results[1:]):
                print "Per-connective stats for stage %s:" % stage.name
                by_connective = eval_metrics.metrics_by_connective()
                remap_by_connective(by_connective)

                for connective, metrics in by_connective.iteritems():
                    csv_metrics = [str(x) for x in connective,
                                   metrics.connective_metrics.tp,
                                   metrics.connective_metrics.fp,
                                   metrics.connective_metrics.fn,
                                   metrics.cause_span_metrics.accuracy,
                                   metrics.cause_head_metrics.accuracy,
                                   metrics.cause_jaccard,
                                   metrics.effect_span_metrics.accuracy,
                                   metrics.effect_head_metrics.accuracy,
                                   metrics.effect_jaccard]
                    print ','.join(csv_metrics)
                print
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
