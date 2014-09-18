""" Define basic pipeline functionality. """

from gflags import DEFINE_list, DEFINE_boolean, DEFINE_integer, FLAGS, DuplicateFlagError
import logging

from util import listify
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
                " for all test paths. If multiple paths are provided, they must"
                " correspond one-to-one with the test paths provided.")
    #DEFINE_boolean('metrics_log_raw_counts', False, "Log raw counts (TP, agreement,"
    #               " etc.) for evaluation or IAA metrics.")
except DuplicateFlagError as e:
    logging.warn('Ignoring redefinition of flag %s' % e.flagname)


class Pipeline(object):
    def __init__(self, stages, reader, writer=None):
        self.stages = listify(stages)
        self.reader = reader
        self.writer = writer
        self.eval_results = None

    def train(self):
        # TODO: set this up to do cross-validation or grid search or whatever.
        # (It should be relatively straightforward using the instances argument
        # to self.test().)
        print "Creating instances..."
        instances = []
        for path in FLAGS.train_paths:
            print 'Reading training data from', path
            self.reader.open(path)
            instances.extend(self.reader.get_all())
        self.reader.close()
        print len(instances), "instances found"

        for stage in self.stages:
            print "Training stage %s..." % stage.name
            stage.train(instances)
            print "Finished training stage", stage.name

    def evaluate(self):
        self.eval_results = []
        self.test()
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
        print 'Testing', batch_size, 'instances at a time'
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

    def test(self, instances=1024):
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
                stage._complete_evaluation(self.eval_results)


class Stage(object):
    def __init__(self, name, models):
        self.name = name
        self.models = models

    def train(self, instances):
        all_parts = []
        for instance in instances:
            all_parts.extend(self._extract_parts(instance))
        assert all_parts, "No parts extracted for training!"
        for model in self.models:
            model.train(all_parts)

    def test(self, instances):
        all_parts = []
        instance_part_counts = [0 for _ in instances]
        parts_by_model = {model.part_type:[] for model in self.models}

        for i in range(len(instances)):
            parts = self._extract_parts(instances[i])
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

    def _evaluate(self, instances):
        raise NotImplementedError

    def _extract_parts(self, instance):
        raise NotImplementedError

    def _decode_labeled_parts(self, instance, labeled_parts):
        raise NotImplementedError

    def _begin_evaluation(self):
        pass

    def _complete_evaluation(self, results):
        pass

    def _prepare_for_evaluation(self, instances):
        pass

class ClassifierStage(Stage):
    def __init__(self, name, models):
        super(ClassifierStage, self).__init__(name, models)
        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0

    def _begin_evaluation(self):
        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0

    def _complete_evaluation(self, results):
        results.append(
            ClassificationMetrics(self.tp, self.fp, self.fn, self.tn))
