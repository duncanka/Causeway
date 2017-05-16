from collections import defaultdict
import colorama
from copy import copy
from gflags import DEFINE_bool, DEFINE_string, FLAGS, DuplicateFlagError
from itertools import chain
import logging
import nltk
from nltk.tag.stanford import StanfordNERTagger
import operator
import os
from os import path
from subprocess import PIPE
import tempfile

from causeway.because_data import CausationInstance
from causeway.because_data.iaa import CausalityMetrics
from nlpypline.pipeline import Stage, Evaluator
from nlpypline.util import listify, print_indented, Enum, make_getter, make_setter

try:
    DEFINE_bool("iaa_calculate_partial", False,
                "Whether to compute metrics for partial overlap")
    DEFINE_string('stanford_ser_path',
                  '/home/jesse/Documents/Work/Research/stanford-ner-2015-04-20',
                  'Path to Stanford NER directory')
    DEFINE_string(
        'stanford_ner_model_name', 'english.all.3class.distsim.crf.ser.gz',
        'Name of model file for Stanford NER')
    DEFINE_bool('print_patterns', False,
                'Whether to print all connective patterns')
    DEFINE_bool('patterns_print_test_instances', False,
                'Whether to print differing IAA results during evaluation of'
                ' pattern matching stage')
    DEFINE_bool('args_print_test_instances', False,
                'Whether to print differing IAA results during evaluation of'
                ' argument labelling stage')
except DuplicateFlagError as e:
    logging.warn('Ignoring flag redefinitions; assuming module reload')


class PossibleCausation(CausationInstance):
    '''
    Acts like a normal CausationInstance object, but with some extra stuff.
    '''

    def __init__(self, sentence, matching_patterns, connective,
                 true_causation_instance=None, cause=None, effect=None,
                 means=None):
        self.sentence = sentence
        self.matching_patterns = listify(matching_patterns)
        self.connective = connective
        self.true_causation_instance = true_causation_instance
        # Cause/effect spans are filled in by the second stage.
        if cause is not None:
            cause = sorted(cause, key=lambda token: token.index)
        self.cause = cause
        if effect is not None:
            cause = sorted(effect, key=lambda token: token.index)
        self.effect = effect
        self.means = means
        self.id = None
        self.type = None
        # TODO: Add spans of plausible ranges for argument spans


class IAAEvaluator(Evaluator):
    # TODO: refactor arguments
    # TODO: provide both pairwise and non-pairwise stats
    def __init__(self, compare_degrees, compare_types, log_test_instances,
                 compare_args, pairwise_only,
                 causations_property_name='causation_instances',
                 log_by_connective=None, log_by_category=None):
        if log_by_connective is None:
            log_by_connective = FLAGS.iaa_log_by_connective
        if log_by_category is None:
            log_by_category = FLAGS.iaa_log_by_category
        self.log_by_connective = log_by_connective
        self.log_by_category = log_by_category
        self.save_differences = bool(log_test_instances or log_by_connective
                                     or log_by_category)

        if FLAGS.iaa_calculate_partial:
            self._with_partial_metrics = CausalityMetrics(
                [], [], True, self.save_differences, None, compare_degrees,
                compare_types, True, True, log_test_instances)
        else:
            self._with_partial_metrics = None
        self._without_partial_metrics = CausalityMetrics(
            [], [], False, self.save_differences, None, compare_degrees,
            compare_types, True, True, log_test_instances)
        self.instances_property_name = causations_property_name
        self.compare_degrees = compare_degrees
        self.compare_types = compare_types
        self.log_test_instances = log_test_instances
        self.compare_args = compare_args
        self.pairwise_only = pairwise_only

    def evaluate(self, document, original_document, sentences,
                 original_sentences):
        if FLAGS.iaa_calculate_partial:
            with_partial = CausalityMetrics(
                original_document.sentences, document.sentences, True,
                self.save_differences, compare_types=self.compare_types,
                compare_args=self.compare_args,
                compare_degrees=self.compare_degrees,
                pairwise_only=self.pairwise_only,
                save_agreements=self.save_differences,
                causations_property_name=self.instances_property_name)
        without_partial = CausalityMetrics(
            original_document.sentences, document.sentences, False,
            self.save_differences, compare_types=self.compare_types,
            compare_args=self.compare_args,
            compare_degrees=self.compare_degrees,
            pairwise_only=self.pairwise_only,
            save_agreements=self.save_differences,
            causations_property_name=self.instances_property_name)

        if self.log_test_instances:
            print 'Differences not allowing partial matches:'
            without_partial.pp(log_stats=False, log_confusion=False,
                               log_differences=True, log_agreements=True,
                               log_by_connective=False, log_by_category=False,
                               indent=1)
            print
            if FLAGS.iaa_calculate_partial:
                print 'Differences allowing partial matches:'
                with_partial.pp(log_stats=False, log_confusion=False,
                                log_differences=True, log_agreements=True,
                                log_by_connective=False, log_by_category=False,
                                indent=1)
                print

        if FLAGS.iaa_calculate_partial:
            self._with_partial_metrics += with_partial
        self._without_partial_metrics += without_partial

    _PERMISSIVE_KEY = 'Allowing partial matches'
    _STRICT_KEY = 'Not allowing partial matches'

    def complete_evaluation(self):
        if FLAGS.iaa_calculate_partial:
            result = {self._PERMISSIVE_KEY: self._with_partial_metrics,
                      self._STRICT_KEY: self._without_partial_metrics}
        else:
            result = self._without_partial_metrics
        self._with_partial_metrics = None
        self._without_partial_metrics = None
        return result

    def aggregate_results(self, results_list):
        if FLAGS.iaa_calculate_partial:
            permissive = results_list[0].aggregate(
                [result_dict[IAAEvaluator._PERMISSIVE_KEY]
                 for result_dict in results_list])
            strict = results_list[0].aggregate(
                [result_dict[IAAEvaluator._STRICT_KEY]
                 for result_dict in results_list])
            return {IAAEvaluator._PERMISSIVE_KEY: permissive,
                    IAAEvaluator._STRICT_KEY: strict}
        else:
            return results_list[0].aggregate(results_list)


# Fix stupid issues with Stanford trying to sentence-split already-split text.
# Unfortunately, the only way to do this without re-running the tagger seems to
# be to split every sentence into a separate file.
class SentenceSplitStanfordNERTagger(StanfordNERTagger):
    def tag_sents(self, sentences):
        encoding = self._encoding
        default_options = ' '.join(nltk.internals._java_options)
        nltk.internals.config_java(options=self.java_options, verbose=False)

        self._input_files = [tempfile.NamedTemporaryFile('w')
                             for _ in sentences]
        for sentence, input_file in zip(sentences, self._input_files):
            _input = ' '.join(sentence)
            if isinstance(_input, nltk.compat.text_type) and encoding:
                _input = _input.encode(encoding)
            input_file.write(_input)
            input_file.flush()

        cmd = list(self._cmd)
        cmd.extend(['-encoding', encoding])

        # Run the tagger and get the output
        stanpos_output, _stderr = nltk.internals.java(
            cmd, classpath=self._stanford_jar, stdout=PIPE, stderr=PIPE)
        stanpos_output = stanpos_output.decode(encoding)

        for input_file in self._input_files:
            input_file.close()
        del self._input_files

        # Return java configurations to their default values
        nltk.internals.config_java(options=default_options, verbose=False)

        return self.parse_output(stanpos_output, sentences)

    @property
    def _cmd(self):
        file_names = ','.join([f.name for f in self._input_files])
        return [
            'edu.stanford.nlp.ie.crf.CRFClassifier', '-loadClassifier',
            self._stanford_model, '-textFiles', file_names,
            '-outputFormat', self._FORMAT, '-tokenizerFactory',
            'edu.stanford.nlp.process.WhitespaceTokenizer', '-tokenizerOptions',
            '\"tokenizeNLs=false\"']


class StanfordNERStage(Stage):
    NER_TYPES = Enum(['Person', 'Organization', 'Location', 'O'])

    def __init__(self, name):
        self.name = name
        # Omit models

    def train(self, documents, instances_by_doc):
        pass

    def _test_documents(self, documents, sentences_by_doc, writer):
        model_path = path.join(FLAGS.stanford_ser_path, 'classifiers',
                               FLAGS.stanford_ner_model_name)
        jar_path = path.join(FLAGS.stanford_ser_path, 'stanford-ner.jar')
        tagger = SentenceSplitStanfordNERTagger(model_path, jar_path)
        tokens_by_sentence = [
            [token.original_text for token in sentence.tokens[1:]]
            for sentence in chain.from_iterable(sentences_by_doc)]
        # Batch process sentences (faster than repeatedly running Stanford NLP)
        ner_results = tagger.tag_sents(tokens_by_sentence)
        all_sentences = chain.from_iterable(sentences_by_doc)
        for sentence, sentence_result in zip(all_sentences, ner_results):
            sentence.tokens[0].ner_tag = None # ROOT has no NER tag
            for token, token_result in zip(sentence.tokens[1:],
                                           sentence_result):
                tag = token_result[1]
                token.ner_tag = self.NER_TYPES.index(tag.title())
            if writer:
                writer.instance_complete(sentence)


def remove_smaller_matches(sentence):
    causations_by_size = defaultdict(list) # causation instances by conn. size
    # Each token can only appear in one connective. Remember which ones have
    # already been deemed part of a larger connective, so that future instances
    # that use that token as part of the connective can be ignored.
    tokens_used = [False for _ in sentence.tokens]
    for causation in sentence.causation_instances:
        causations_by_size[len(causation.connective)].append(causation)

    causations_to_keep = []
    # Process connectives biggest to smallest, discarding any that reuse tokens.
    # If we have two connectives of the same length competing for a token, this
    # will arbitrarily choose the first one we find.
    for connective_length in sorted(causations_by_size.keys(), reverse=True):
        for causation in causations_by_size[connective_length]:
            for conn_token in causation.connective:
                if tokens_used[conn_token.index]:
                    break
            else: # Executes only if loop over tokens didn't break
                causations_to_keep.append(causation)
                for conn_token in causation.connective:
                    tokens_used[conn_token.index] = True

    sentence.causation_instances = causations_to_keep


RELATIVE_POSITIONS = Enum(['Before', 'Overlapping', 'After'])


def get_causation_tuple(connective_tokens, cause_head, effect_head):
    return (tuple(t.index for t in connective_tokens),
            cause_head.index if cause_head else None,
            effect_head.index if effect_head else None)


# Add some Colorama functionality.
for style, code in [('UNDERLINE', 4), ('BLINK', 5)]:
    setattr(colorama.Style, style, '\033[%dm' % code)
