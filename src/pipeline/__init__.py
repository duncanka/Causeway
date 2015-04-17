""" Define basic pipeline functionality. """

from gflags import DEFINE_list, DEFINE_boolean, DEFINE_integer, FLAGS, DuplicateFlagError
from copy import deepcopy
import itertools
import logging
from numpy import random

from util import listify, partition, print_indented
from util.metrics import ClassificationMetrics

# Define pipeline-related flags.
try:
    DEFINE_boolean('evaluate', False,
                   "True for evaluating the parser (requires --test_paths).")
    DEFINE_list('train_paths', [],
                "Paths to the files containing the training data")
    DEFINE_list('test_paths', [],
                "Paths to the files containing the test data")
    DEFINE_list('test_output_paths', [],
                "Paths at which to place the test results."
                " Defaults to test_paths. If this is a single path, it is used"
                " for all test paths. If multiple paths are provided, they"
                " must correspond one-to-one with the test paths provided.")
    DEFINE_integer('cv_folds', 10,
                   'How many folds to split data into for cross-validation. A'
                   ' negative value indicates leave-one-out CV.')
    DEFINE_integer('cv_debug_stop_after', None,
                   'Number of CV rounds to stop after (for debugging)')
    DEFINE_integer('test_batch_size', 1024, 'Batch size for testing.')
    DEFINE_boolean('cv_print_fold_results', True,
                   "Whether to print each fold's results as they are computed")
except DuplicateFlagError as e:
    logging.warn('Ignoring redefinition of flag %s' % e.flagname)


class Pipeline(object):
    def __init__(self, stages, reader, writer=None):
        self.stages = listify(stages)
        self.reader = reader
        self.writer = writer
        self.eval_results = None

    def _read_instances(self, paths=None):
        logging.info("Creating instances...")
        instances = []
        for path in paths:
            logging.info('Reading instances from %s' % path)
            self.reader.open(path)
            instances.extend(self.reader.get_all())
        self.reader.close()
        return instances

    def cross_validate(self, num_folds=None, stage_aggregators=None):
        '''
        Returns a list of results, organized by stage. If stage_aggregators is
        provided, each list item is an aggregate result for the corresponding
        stage. Otherwise, each list item is a list of the results achieved on
        each fold.

        stage_aggregators is a list of functions, each of which takes a list of
        results for the corresponding pipeline stage (one result per fold) and
        aggregates them into a single result for the stage.
        '''
        if stage_aggregators:
            assert len(stage_aggregators) == len(self.stages)
        if num_folds is None:
            num_folds = FLAGS.cv_folds

        instances = self._read_instances(FLAGS.train_paths + FLAGS.test_paths)
        random.shuffle(instances)
        if num_folds < 0:
            num_folds = len(instances)
        folds = partition(instances, num_folds)
        results = [[] for _ in self.stages]

        for i, fold in enumerate(folds):
            print "Beginning fold", i + 1, 'of', num_folds
            testing = fold
            training = list(itertools.chain(
                *[f for f in folds if f is not fold]))
            self.train(training)

            # Copy testing data so we don't overwrite original instances.
            fold_results = self.evaluate(deepcopy(testing))

            if FLAGS.cv_print_fold_results:
                print "Fold", i + 1, "results:"

            for stage, stage_results, current_stage_result in zip(
                self.stages, results, fold_results):
                stage_results.append(current_stage_result)
                if FLAGS.cv_print_fold_results:
                    print_indented(1, 'Stage', stage.name, 'results:')
                    self.print_stage_results(2, current_stage_result)

            if (FLAGS.cv_debug_stop_after is not None
                and i + 1 >= FLAGS.cv_debug_stop_after):
                break

        if stage_aggregators:
            results = [aggregator(stage_results)
                       for aggregator, stage_results
                        in zip(stage_aggregators, results)]
        self.eval_results = results
        return results

    @staticmethod
    def print_stage_results(indent_baseline, results, result_names=[]):
        # TODO: Make result names something a stage can define for itself
        if isinstance(results, list) or isinstance(results, tuple):
            for i, r in enumerate(results):
                try:
                    result_name = result_names[i]
                except IndexError:
                    result_name = 'Evaluation result %d' % i
                print_indented(indent_baseline, result_name, ':', sep='')
                print_indented(indent_baseline + 1, str(r))
        else:
            print_indented(indent_baseline, results)

    def train(self, instances=None):
        if instances is None:
            instances = self._read_instances(FLAGS.train_paths)
            logging.info("%d instances found" % len(instances))
        else:
            logging.info("Training on %d instances" % len(instances))

        for stage in self.stages:
            print "Training stage %s..." % stage.name
            stage.train(instances)
            print "Finished training stage", stage.name

    def evaluate(self, instances=FLAGS.test_batch_size):
        self.eval_results = []
        self.test(instances)
        results = self.eval_results
        self.eval_results = None
        return results

    def __test_instances(self, instances):
        for stage in self.stages:
            if self.eval_results is not None:
                stage._prepare_for_evaluation(instances)
            stage.test(instances)
            if self.eval_results is not None:
                stage._evaluate(instances)

    def __set_up_paths(self):
        if not FLAGS.test_output_paths:
            FLAGS.test_output_paths = FLAGS.test_paths
        else:
            if len(FLAGS.test_output_paths) == 1:
                FLAGS.test_output_paths = [FLAGS.test_output_paths[0] for _
                                           in range(len(FLAGS.test_paths))]
            else:
                assert (len(FLAGS.test_paths) == len(FLAGS.test_output_paths)
                        ), ("Test path count & test output path count conflict")

    def __test_instances_from_reader(self, batch_size):
        logging.info('Testing %d instances at a time' % batch_size)
        if (not self.writer):
            logging.warn("No writer provided; pipeline results not written"
                         " anywhere")

        for path, output_path in zip(FLAGS.test_paths,
                                     FLAGS.test_output_paths):
            self.reader.open(path)
            if self.writer:
                self.writer.open(output_path)

            last_batch = False
            while not last_batch:
                instances = []
                for _ in xrange(batch_size):
                    next_instance = self.reader.get_next()
                    if not next_instance:
                        last_batch = True
                        break
                    instances.append(next_instance)
                self.__test_instances(instances)
                if self.writer:
                    self.writer.write(instances)

        if self.writer:
            self.writer.close()

    def test(self, instances=FLAGS.test_batch_size):
        """
        If instances is an integer, instances are read from the files specified
        in the test_path flags, and the parameter is interpreted as the number
        of instances to read/process per batch. If instances is a list, it is
        used instead of reading instances from files.
        """
        if self.eval_results is not None:
            for stage in self.stages:
                stage._begin_evaluation()

        if isinstance(instances, list):
            print 'Testing', len(instances), 'instances'
            self.__test_instances(instances)
        else: # it's an int
            batch_size = instances
            self.__set_up_paths()
            self.__test_instances_from_reader(batch_size)

        if self.eval_results is not None:
            for stage in self.stages:
                self.eval_results.append(stage._complete_evaluation())


class Stage(object):
    def __init__(self, name, models):
        self.name = name
        self.models = models

    def train(self, instances):
        all_parts = []
        for instance in instances:
            all_parts.extend(self._extract_parts(instance, True))
            # In general, consumed attributes are only used for part extraction.
            # If they are needed for some reason in further processing, the
            # relevant information should simply be attached to the parts.
            self.__consume_attributes(instance)

        assert all_parts, "No parts extracted for training!"
        for model in self.models:
            model.train(all_parts)

    def test(self, instances):
        all_parts = []
        instance_part_counts = [0 for _ in instances]
        parts_by_model = {model.part_type:[] for model in self.models}

        for i, instance in enumerate(instances):
            parts = self._extract_parts(instance, False)
            self.__consume_attributes(instance)
            all_parts.extend(parts)
            instance_part_counts[i] = len(parts)
            for part in parts:
                # Throws an exception if any parts aren't handled by some model.
                parts_by_model[type(part)].append(part)

        for model in self.models:
            parts = parts_by_model[model.part_type]
            model.test(parts)

        parts_processed = 0
        for instance, part_count in zip(instances, instance_part_counts):
            next_parts_processed = parts_processed + part_count
            self._decode_labeled_parts(
                instance, all_parts[parts_processed:next_parts_processed])
            parts_processed = next_parts_processed

    def __consume_attributes(self, instance):
        for attribute_name in self.CONSUMED_ATTRIBUTES:
            logging.debug('Consuming', attribute_name)
            delattr(instance, attribute_name)

    '''
    Default list of attributes the stage adds to instances. Add a class-wide
    field by the same name in the class for a stage that adds any attributes
    to instances.
    '''
    PRODUCED_ATTRIBUTES = []

    '''
    Default list of attributes the stage removes from instances. Add a
    class-wide field by the same name in the class for a stage that removes any
    attributes from instances.
    '''
    CONSUMED_ATTRIBUTES = []

    def _evaluate(self, instances):
        raise NotImplementedError

    def _extract_parts(self, instance, is_train):
        raise NotImplementedError

    def _decode_labeled_parts(self, instance, labeled_parts):
        pass

    def _begin_evaluation(self):
        pass

    def _complete_evaluation(self):
        pass

    def _prepare_for_evaluation(self, instances):
        pass

class ClassifierStage(Stage):
    def __init__(self, name, models, *args, **kwargs):
        super(ClassifierStage, self).__init__(name=name, models=models,
                                              *args, **kwargs)
        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0

    def _begin_evaluation(self):
        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0

    def _complete_evaluation(self):
        return ClassificationMetrics(self.tp, self.fp, self.fn, self.tn)
