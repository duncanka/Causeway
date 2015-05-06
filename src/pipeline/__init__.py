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
    def __init__(self, stages, reader, writer=None, copy_fn=deepcopy):
        '''
        copy_fn is the function to use when copying over instances to avoid
        lasting impacts of modification during testing. Defaults to built-in
        deep copy; providing an alternative function can make copying faster if
        not all elements need to be copied to protect originals from
        modification. Clients should make certain that the copy function
        provided duplicates all instance properties that may be modified by
        testing at *any point* in the pipeline. 
        '''
        self.stages = listify(stages)
        self.reader = reader
        self.writer = writer
        self._evaluating = False
        self._copy_fn = copy_fn

    def _read_instances(self, paths=None):
        logging.info("Creating instances...")
        instances = []
        for path in paths:
            logging.info('Reading instances from %s' % path)
            self.reader.open(path)
            instances.extend(self.reader.get_all())
        self.reader.close()
        return instances

    def cross_validate(self, num_folds=None):
        '''
        Returns a list of results, organized by stage. Results are also saved in
        self.eval_results. Results are aggregated across all folds using the
        aggregator function provided by the stage (see
        Stage.aggregate_eval_results).
        '''
        if num_folds is None:
            num_folds = FLAGS.cv_folds

        logging.info("Evaluating with %d-fold cross-validation" % num_folds)
        if FLAGS.cv_debug_stop_after:
            logging.info('(Stopping after %d fold%s)'
                         % (FLAGS.cv_debug_stop_after,
                            '' if FLAGS.cv_debug_stop_after == 1 else 's'))

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

            # Evaluation copies the data for testing, so we don't have to worry
            # about overwriting our original instances.
            fold_results = self.evaluate(testing)

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

        results = [stage.aggregate_eval_results(stage_results)
                   for stage, stage_results in zip(self.stages, results)]
        self.eval_results = results
        return results

    @staticmethod
    def print_stage_results(indent_baseline, results):
        if isinstance(results, dict): # special case: treat as named sub-metrics
            for result_name, result in results.items():
                print_indented(indent_baseline, result_name, ':', sep='')
                print_indented(indent_baseline + 1, str(result))
        else:
            print_indented(indent_baseline, results)

    def print_eval_results(self, eval_results, indent_baseline=0):
        for stage, result in zip(self.stages, eval_results):
            print_indented(indent_baseline, "Evaluation for stage ",
                           stage.name, ':', sep='')
            self.print_stage_results(indent_baseline + 1, result)

    def train(self, instances=None):
        '''
        Trains all stages in the pipeline. Each stage is trained on instances
        on which the previous stage has been tested on, so that it sees a
        realistic view of what its inputs will look like. The instance objects
        provided in instances are not modified in this process.
        '''
        if instances is None:
            instances = self._read_instances(FLAGS.train_paths)
            logging.info("%d instances found" % len(instances))
        else:
            # Copy over instances so that when test() is called below, it won't
            # overwrite the originals. This is especially important during
            # cross-validation, when we could otherwise overwrite data that will
            # later be used again for both training and testing.
            if len(self.stages) > 1:
                instances = [self._copy_fn(instance) for instance in instances]
                logging.info("Training on %d instances" % len(instances))

        for stage in self.stages:
            logging.info("Training stage %s..." % stage.name)
            stage.train(instances)
            logging.info("Finished training stage %s" % stage.name)
            # For training, each stage needs a realistic view of what its inputs
            # will look like. So now that we've trained the stage, if there is
            # another stage after it we run the trained stage as though we were
            # in testing mode.
            # TODO: Should we allow disabling this somehow?
            if stage is not self.stages[-1]:
                logging.info("Testing stage %s for input to next stage..."
                             % stage.name)
                stage.test(instances)
                logging.info("Done testing stage %s" % stage.name)

    def evaluate(self, instances=FLAGS.test_batch_size):
        '''
        Evaluates a single batch of instances. Returns evaluation metrics.
        '''
        eval_results = []
        self._evaluating = True
        for stage in self.stages:
            stage._begin_evaluation()

        self.test(instances)

        for stage in self.stages:
            eval_results.append(stage._complete_evaluation())
        self._evaluating = False

        return eval_results

    def __test_instances(self, instances):
        original_instances = instances
        if self._evaluating:
            instances = [self._copy_fn(instance) for instance in instances]
        for stage in self.stages:
            stage.test(instances)
            if self._evaluating:
                stage._evaluate(instances, original_instances)

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
        used instead of reading instances from files, and the original instances
        are modified during testing. (Note that the original instance are NOT
        modified during evaluation.)
        """

        if isinstance(instances, list):
            print 'Testing', len(instances), 'instances'
            self.__test_instances(instances)
        else: # it's an int
            batch_size = instances
            self.__set_up_paths()
            self.__test_instances_from_reader(batch_size)

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

    @staticmethod
    def aggregate_eval_results(results_list):
        '''
        Aggregates a list of evaluation results, e.g., from cross-validation.
        Should generally do some kind of averaging. By default this just returns
        the original list of results; if any kind of processing should be done,
        this method must be overridden.
        '''
        return results_list

    def _evaluate(self, instances, original_instances):
        '''
        Evaluates a single batch of instances. original_instances is the
        list of instances unmodified by testing, and from which instances were
        copied before testing.
        '''
        raise NotImplementedError

    def _extract_parts(self, instance, is_train):
        raise NotImplementedError

    def _decode_labeled_parts(self, instance, labeled_parts):
        pass

    def _begin_evaluation(self):
        pass

    def _complete_evaluation(self):
        '''
        Should return the evaluation results for this stage, incorporating the
        results of all calls to self._evaluate. If a dict is returned, it will
        be treated as a collection of named result metrics, where each key
        indicates the name of the corresponding metric.
        '''
        raise NotImplementedError

class ClassifierStage(Stage):
    def __init__(self, name, models, *args, **kwargs):
        super(ClassifierStage, self).__init__(name=name, models=models,
                                              *args, **kwargs)
        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0

    @staticmethod
    def aggregate_eval_results(results_list):
        return ClassificationMetrics.average(results_list)

    def _begin_evaluation(self):
        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0

    def _complete_evaluation(self):
        return ClassificationMetrics(self.tp, self.fp, self.fn, self.tn)
