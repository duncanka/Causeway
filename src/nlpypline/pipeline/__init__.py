""" Define basic pipeline functionality. """

from copy import deepcopy
from gflags import (DEFINE_list, DEFINE_boolean, DEFINE_integer, FLAGS,
                    DuplicateFlagError, DEFINE_string, FlagsError)
import itertools
import logging
import numpy as np
import os
from os import path
import sys

from nlpypline.data import SentencesDocument
from nlpypline.pipeline.models import SimpleModel, Model
from nlpypline.util import listify, partition, print_indented
from nlpypline.util.metrics import ClassificationMetrics
from nlpypline.util.freezer import freeze


# Define pipeline-related flags.
try:
    DEFINE_boolean('evaluate', False,
                   "True for evaluating the pipeline (requires --test_paths).")
    DEFINE_list('train_paths', [],
                "Paths to the files containing the training data")
    DEFINE_list('test_paths', [],
                "Paths to the files containing the test data")
    DEFINE_list('test_output_paths', [],
                "Directories in which to place the test results (filenames will"
                " be based on the input filenames). Defaults to the directories"
                " from  test_paths. If this is a single path, it is used for"
                " all test paths. If multiple paths are provided, they must"
                " correspond one-to-one with the test paths provided.")
    DEFINE_string('test_output_extension', 'predicted',
                  'Extension to add to the output filenames.')
    DEFINE_integer('cv_folds', 10,
                   'How many folds to split data into for cross-validation. A'
                   ' 0 or negative value indicates leave-one-out CV.')
    DEFINE_integer('cv_debug_stop_after', None,
                   'Number of CV rounds to stop after (for debugging)')
    DEFINE_integer('cv_debug_start_at', 1,
                   'CV round to start at (1-indexed; for debugging)')
    DEFINE_boolean('cv_print_fold_results', True,
                   "Whether to print each fold's results as they are computed")
    DEFINE_boolean('cv_by_sentences', True,
                   "Whether CV folds are split by sentences, rather than by"
                   " documents. Only valid if documents are SentencesDocuments."
                   " If True, folds are split by creating new pseudo-documents"
                   " with randomly partitioned instances.")
    DEFINE_integer('test_doc_batch_size', 1,
                   "If documents are being read from test_paths, defines how"
                   " many test documents are read and tested at once. -1 means"
                   " read all documents at once.")
    DEFINE_boolean('eval_training_test', False,
                   'When testing a stage at training time (usually to give the'
                   ' next stage realistic input), evaluation results will be'
                   ' printed if this stage is true. Causes all stages to'
                   ' default to being tested at train time on training data.')
except DuplicateFlagError as e:
    logging.warn('Ignoring flag redefinitions; assuming module reload')


class Pipeline(object):
    # TODO: allow creating a Pipeline with no reader, which can only be used
    # with documents passed in as function arguments.
    def __init__(self, stages, reader, writer=None):
        '''
        copy_fn is the function to use when copying over documents to avoid
        lasting impacts of modification during testing. Defaults to built-in
        deep copy; providing an alternative function can make copying faster if
        not all elements need to be copied to protect originals from
        modification. Clients should make certain that the copy function
        provided duplicates all document properties that may be modified by
        testing at *any point* in the pipeline.
        '''
        self.stages = listify(stages)
        # Disallow duplicate stage names, which would mess with model-saving.
        all_names = set()
        for stage in self.stages:
            if stage.name in all_names:
                raise ValueError("Duplicate stage name: %s" % stage.name)
            all_names.add(stage.name)

        self.reader = reader
        self.writer = writer
        self._evaluators_by_stage = []

        # During training, test all stages whose successors might need realistic
        # training inputs. That means we can exclude the final stage, as well as
        # any stages after which the only training operation is the default,
        # i.e., a no-op.
        self.test_stages_during_training_mask = np.full((len(stages)), True,
                                                        bool)
        # if evaluating training tests, leave all defaults as True.
        if not FLAGS.eval_training_test:
            self.test_stages_during_training_mask[-1] = False
            for i in range(len(stages) - 2, -1, -1): # iterate from penultimate
                next_stage = stages[i + 1]
                if (next_stage.model.__class__._train_model
                    == Model._train_model
                    and next_stage.__class__.train == Stage.train):
                    self.test_stages_during_training_mask[i] = False
                else:
                    # This stage and all before it should test during training.
                    break

    def _read_documents(self, doc_paths=None):
        documents = []
        for doc_path in doc_paths:
            logging.info('Reading documents from %s' % doc_path)
            self.reader.open(doc_path)
            documents.extend(self.reader.get_all())
        self.reader.close()
        return documents

    @staticmethod
    def weight_doc_by_sentences(document): # a common document weight function
        return len(document.sentences)

    def cross_validate(self):
        '''
        Returns a list of results, organized by stage. Results are also saved in
        self.eval_results. Results are aggregated across all folds using the
        aggregator functions provided by the stage's evaluators (see
        Evaluator.aggregate_results).
        '''
        num_folds = FLAGS.cv_folds

        logging.info("Evaluating with %d-fold cross-validation" % num_folds)
        if FLAGS.cv_debug_stop_after:
            logging.info('(Stopping after %d fold%s)'
                         % (FLAGS.cv_debug_stop_after,
                            '' if FLAGS.cv_debug_stop_after == 1 else 's'))

        documents = self._read_documents(FLAGS.train_paths + FLAGS.test_paths)
        if FLAGS.cv_by_sentences:
            # TODO: does this need to allow CV by any other kind of instance?
            if not isinstance(documents[0], SentencesDocument):
                raise FlagsError("Can't use cv_by_sentences when documents are"
                                 " not SentencesDocuments")
            all_instances = list(itertools.chain(*[d.sentences
                                                   for d in documents]))
            print len(all_instances), "instances"
            np.random.shuffle(all_instances)
            if num_folds <= 0:
                num_folds = len(all_instances)
            sentence_folds = partition(all_instances, num_folds)
            folds = [[SentencesDocument('fold%d' % i, sentences)]
                     for i, sentences in enumerate(sentence_folds)]
        else:
            np.random.shuffle(documents)
            if num_folds <= 0:
                num_folds = len(documents)
            folds = partition(documents, num_folds)

        results = [[] for _ in self.stages] # each list holds results by fold

        for i, fold in list(enumerate(folds))[FLAGS.cv_debug_start_at - 1:]:
            print "Beginning fold", i + 1, 'of', num_folds
            testing = fold
            training = list(itertools.chain(
                *[f for f in folds if f is not fold]))
            self.train(training)

            # Evaluation copies the data for testing, so we don't have to worry
            # about overwriting our original documents.
            fold_results = self.evaluate(testing)

            if FLAGS.cv_print_fold_results:
                print "Fold", i + 1, "results:"

            for stage, stage_results, stage_fold_result in zip(
                self.stages, results, fold_results):
                stage_results.append(stage_fold_result)
                if FLAGS.cv_print_fold_results and None != stage_fold_result:
                    print_indented(1, 'Stage "', stage.name, '" results:',
                                   sep='')
                    self.print_stage_results(2, stage_fold_result)
            sys.stdout.flush() # make stage results visible immediately

            if (FLAGS.cv_debug_stop_after is not None
                and i + 1 >= FLAGS.cv_debug_stop_after):
                break

        evaluators = [stage._make_evaluator() for stage in self.stages]
        results = [(stage_evaluator.aggregate_results(stage_results)
                    if stage_evaluator else None)
                   for stage_evaluator, stage_results in zip(
                       evaluators, results)]
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
            if result is not None:
                print_indented(indent_baseline, 'Evaluation for stage "',
                               stage.name, '":', sep='')
                self.print_stage_results(indent_baseline + 1, result)

    def train(self, documents=None):
        '''
        Trains all stages in the pipeline. Each stage is trained on documents
        on which the previous stage has been tested on, so that it sees a
        realistic view of what its inputs will look like. The elements of the
        documents argument, if provided, are not modified in this process.
        '''
        if documents is None:
            documents = self._read_documents(FLAGS.train_paths)
            logging.info("%d documents found" % len(documents))
        else:
            # Copy over documents so that when test() is called below, it won't
            # overwrite the originals. This is especially important during
            # cross-validation, when we could otherwise overwrite data that will
            # later be used again for both training and testing.
            documents = [deepcopy(document) for document in documents]
            logging.info("Training on %d documents" % len(documents))

        for i, stage in enumerate(self.stages):
            logging.info('Training stage "%s"...' % stage.name)
            instances_by_document = [
                stage._extract_instances(doc, True, False) for doc in documents]
            stage.train(documents, instances_by_document)
            logging.info('Finished training stage "%s"' % stage.name)

            # For training, each stage needs a realistic view of what its inputs
            # will look like. So now that we've trained the stage, if there is
            # another trainable stage after it we run the trained stage as
            # though we were in testing mode.
            # ...or at least that's the default behavior. This can be overridden
            # for particular pipelines by changing
            # test_stages_during_training_mask.
            if self.test_stages_during_training_mask[i]:
                self._do_training_test(stage, documents)

            # TODO: Fix attribute consumption
            else: # consume attributes because it won't happen via test()
                for document, instances in zip(documents,
                                               instances_by_document):
                    stage._consume_attributes(document, instances)

    def _do_training_test(self, stage, documents):
        logging.info('Testing stage "%s" for input to next stage...'
                             % stage.name)
        instances_by_doc = [
            stage._extract_instances(document, False, False)
            for document in documents]

        if FLAGS.eval_training_test:
            evaluator = stage._make_evaluator()
            original_docs = deepcopy(documents)
            original_instances_by_doc = deepcopy(instances_by_doc)

        stage._test_documents(documents, instances_by_doc, None)

        if FLAGS.eval_training_test and evaluator is not None:
            for (document, original_doc, instances, original_instances) in zip(
                    documents, original_docs, instances_by_doc,
                    original_instances_by_doc):
                evaluator.evaluate(document, original_doc, instances,
                                   original_instances)
            print "Training results for stage %s:" % stage.name
            self.print_stage_results(1, evaluator.complete_evaluation())

        logging.info('Done testing stage "%s"' % stage.name)

    def evaluate(self, documents=None):
        '''
        Evaluates the pipeline on a collection of documents (by running the
        pipeline and then letting the stages' evaluators compare the results to
        the originals, which are assumed to have gold-standard labels). Returns
        a list of evaluation metrics by stage.
        '''
        self._evaluators_by_stage = [stage._make_evaluator()
                                     for stage in self.stages]
        self.test(documents)
        eval_results = [evaluator.complete_evaluation() if evaluator else None
                        for evaluator in self._evaluators_by_stage]

        self._evaluators_by_stage = []
        return eval_results

    def save_models(self, directory):
        if not path.isdir(directory):
            os.makedirs(directory)
        for stage in self.stages:
            filename = path.join(directory, "%s.model" % stage.name)
            stage.model.save(filename)
        logging.info("All models saved to %s", directory)

    def load_models(self, directory):
        for stage in self.stages:
            filename = path.join(directory, "%s.model" % stage.name)
            logging.info("Loading model %s...", filename)
            stage.model.load(filename)
        logging.info("Loaded all models.")

    def __test_documents(self, documents):
        if self._evaluators_by_stage: # we're evaluating; avoid overwriting
            original_documents = freeze(documents) # freeze just as a precaution
            documents = [deepcopy(doc) for doc in documents]

        for i, stage in enumerate(self.stages):
            if self._evaluators_by_stage:
                original_instances_by_doc = [
                    stage._extract_instances(original_document, False, True)
                    for original_document in original_documents]

            instances_by_doc = [stage._extract_instances(document, False, False)
                                for document in documents]

            logging.info('Testing stage "%s"...' % stage.name)

            # On the final stage, the instance is now complete, so provide a
            # writer (if we have one).
            writer = self.writer if i == len(self.stages) - 1 else None
            stage._test_documents(documents, instances_by_doc, writer)

            if self._evaluators_by_stage:
                for (document, original_document, instances, original_instances
                     ) in zip(documents, original_documents, instances_by_doc,
                              original_instances_by_doc):
                    if self._evaluators_by_stage[i]:
                        self._evaluators_by_stage[i].evaluate(
                            document, original_document, instances,
                            original_instances)

    def __set_up_paths(self):
        if not FLAGS.test_output_paths:
            FLAGS.test_output_paths = [path.dirname(test_path)
                                       for test_path in FLAGS.test_paths]
        else:
            if len(FLAGS.test_output_paths) == 1:
                FLAGS.test_output_paths = [FLAGS.test_output_paths[0] for _
                                           in range(len(FLAGS.test_paths))]
            else:
                assert (len(FLAGS.test_paths) == len(FLAGS.test_output_paths)
                        ), ("Test path count & test output path count conflict")

    def __test_documents_from_reader(self):
        if (not self.writer):
            logging.warn("No writer provided; pipeline results not being"
                         " written anywhere")

        paths_written = set()
        for input_path, output_path in zip(FLAGS.test_paths,
                                           FLAGS.test_output_paths):
            print 'Testing files from %s...' % input_path
            self.reader.open(input_path)
            if self.writer:
                base_file_name = path.splitext(path.basename(input_path))[0]
                output_file_name = '.'.join([base_file_name,
                                             FLAGS.test_output_extension])
                output_file_path = path.join(output_path, output_file_name)

                if output_file_path in paths_written:
                    self.writer.open(output_file_path, 'a')
                else:
                    paths_written.add(output_file_path)
                    self.writer.open(output_file_path, 'w')

            if FLAGS.test_doc_batch_size > 0:
                if FLAGS.test_doc_batch_size != 1:
                    logging.info("Processing documents in batches of %d",
                                 FLAGS.test_doc_batch_size)
                doc_groups = itertools.izip_longest(
                    *([iter(self.reader)] * FLAGS.test_doc_batch_size))
            else:
                logging.info("Processing all documents in one batch")
                doc_groups = [list(self.reader)]
            for documents in doc_groups:
                if documents[-1] is None: # padded by izip_longest
                    documents = [d for d in documents if d is not None]
                self.__test_documents(documents)
                if self.writer:
                    for document in documents:
                        self.writer.write(document)

        if self.writer:
            self.writer.close()

    def test(self, documents=None):
        """
        If a documents list is provided, it is used instead of reading documents
        from files, and the original documents are modified during testing.
        (Note that the original documents are NOT modified during evaluation.)
        Otherwise, documents are read from the files specified in the test_path
        flags, and the results are written by the pipeline's Writer (if it has
        one).
        """

        if documents is not None:
            print 'Testing', len(documents), 'documents'
            self.__test_documents(documents)
        else:
            self.__set_up_paths()
            self.__test_documents_from_reader()


class Evaluator(object):
    '''
    Base class for evaluating models. Assumes that the instances extracted by
    each stage for its model are all that's of interest for evaluation.
    '''

    def evaluate(self, document, original_document, instances,
                 original_instances):
        '''
        Evaluates a batch of instances. original_instances is the list of
        instances unmodified by testing, and from which instances were copied
        before testing.
        '''
        raise NotImplementedError

    def complete_evaluation(self):
        '''
        Should return the evaluation results for this stage, incorporating the
        results of all calls to self.evaluate. If a dict is returned, it will
        be treated as a collection of named result metrics, where each key
        indicates the name of the corresponding metric.
        '''
        raise NotImplementedError

    def aggregate_results(self, results_list):
        '''
        Aggregates a list of evaluation results, e.g., from cross-validation.
        Should generally do some kind of averaging. By default this just returns
        the original list of results; if any kind of processing should be done,
        this method must be overridden.
        '''
        return results_list
# TODO: Is it worth it to define a default compound evaluator (i.e., an
# evaluator that groups two other named evaluators)?


class Stage(object):
    def __init__(self, name, model):
        self.name = name
        self.model = model

    def train(self, documents, instances_by_document):
        '''
        In general, the instances should encapsulate everything the model needs
        to know to train, but just in case, the Stage is provided with
        information on what documents the instances came from. Stages can
        override this function to make use of this information.
        '''
        self.model.train(list(itertools.chain(*instances_by_document)))

    def test(self, document, instances, writer=None):
        predicted_outputs = self.model.test(instances)
        if predicted_outputs is not None:
            for instance, predicted_output in itertools.izip_longest(
                instances, predicted_outputs):
                self._label_instance(document, instance, predicted_output)
                if writer:
                    writer.instance_complete(document, instance)
        elif writer: # No additional labeling needed; signal instance completion
            for instance in instances:
                writer.instance_complete(document, instance)

    def _prepare_document(self, document, instances):
        '''
        Prepares the document for testing (e.g., setting variables on the
        document that will be harder to set after processing the instances).
        '''
        pass

    def _document_complete(self, document):
        '''
        Handles any post-labeling processing for the entire document.
        '''
        pass

    # TODO: rename to "get_evaluator"
    def _make_evaluator(self):
        '''
        Creates a new Evaluator object that knows how to properly evaluate
        documents for this stage. Must be overridden for any stage that supports
        evaluation.
        '''
        return None

    def _consume_attributes(self, document, instances):
        # TODO: fix me
        for attribute_name in self.consumed_attributes:
            for instance in instances:
                delattr(instance, attribute_name)

    def _test_documents(self, documents, instances_by_doc, writer):
        '''
        In the vast majority of cases, stages should process document by
        document, using the test() function above. However, it may occasionally
        be necessary to override this functionality to batch-process documents
        (e.g., for efficiency reasons). In such cases, the overridden method
        should be certain to call _label_instance and writer.instance_complete
        as appropriate.
        '''
        for document, instances in zip(documents, instances_by_doc):
            self._prepare_document(document, instances)
            self.test(document, instances, writer)
            self._document_complete(document)

    '''
    Default list of attributes the stage adds to instances. Add a class-wide
    field by the same name in the class for a stage that adds any attributes
    to instances. Stages can override with either class-level or instance-level
    lists.
    '''
    produced_attributes = []

    '''
    Default list of attributes the stage removes from instances. Add a
    class-wide field by the same name in the class for a stage that removes any
    attributes from instances. Stages can also provide instance-specific lists.
    '''
    consumed_attributes = []

    def _extract_instances(self, document, is_train, is_original):
        '''
        Sentences are the most commonly used unit of analysis for models, so
        the default is to return sentences as instances.
        For other document types or to implement other behavior, this method
        must be overridden.
        '''
        if hasattr(document, 'sentences'): # Will work on SentencesDocument
            return document.sentences
        else:
            raise NotImplementedError

    def _label_instance(self, document, instance, predicted_output):
        pass


class SimpleStage(Stage):
    '''
    A stage that simply runs an operation on each instance, with the operation
    encapsulated as a single function. The function should be entirely
    self-contained; its return value will be ignored.
    
    If instances other than the default are required, this stage should be
    overridden with an appropriate _extract_instances function.
    '''
    def __init__(self, name, operation, evaluator_fn=lambda: None):
        super(SimpleStage, self).__init__(name, SimpleModel(operation))
        self._evaluator_fn = evaluator_fn

    def _make_evaluator(self):
        return self._evaluator_fn()


class ClassifierEvaluator(Evaluator):
    def __init__(self):
        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0

    def aggregate_results(self, results_list):
        return ClassificationMetrics.average(results_list)

    def complete_evaluation(self):
        return ClassificationMetrics(self.tp, self.fp, self.fn, self.tn)
