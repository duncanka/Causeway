from gflags import DEFINE_bool, FLAGS, DuplicateFlagError
import logging

from causality_pipelines import PossibleCausation, IAAEvaluator
from pipeline import Stage
from pipeline.models import Model

try:
    DEFINE_bool('combiner_print_test_instances', False,
                'Whether to print differing IAA results during evaluation')
except DuplicateFlagError as e:
    logging.warn('Ignoring redefinition of flag %s' % e.flagname)


class BaselineCombinerModel(Model):
    def __init__(self):
        super(BaselineCombinerModel, self).__init__(PossibleCausation)

    def train(self, parts):
        pass

    @staticmethod
    def _get_instance_tuple(causation_instance, sentence):
        arg_indices = tuple(
            sentence.get_head(arg) if arg else None
            for arg in [causation_instance.cause, causation_instance.effect])
        connective_indices = tuple(t.index for t in
                                   causation_instance.connective)
        return arg_indices + connective_indices

    def test(self, parts):
        # TODO: make this avoid preserving two annotations that differ only in
        # that one of them is missing an argument. Particularly, make it prefer
        # the one that has an argument.
        last_sentence = None
        existing_causations_in_sentence = None
        for possible_causation in parts:
            sentence = possible_causation.sentence
            if sentence is not last_sentence:
                existing_causations_in_sentence = []
                for causation in sentence.causation_instances:
                    existing_causations_in_sentence.append(
                        self._get_instance_tuple(causation, sentence))
                
            if (self._get_instance_tuple(possible_causation, sentence)
                not in existing_causations_in_sentence):
                possible_causation.sentence.add_causation_instance(
                    connective=possible_causation.connective,
                    cause=possible_causation.cause,
                    effect=possible_causation.effect)


class BaselineCombinerStage(Stage):
    def __init__(self, name, baseline_causations_attr_name):
        super(BaselineCombinerStage, self).__init__(name,
                                                    BaselineCombinerModel())
        self.consumed_attributes = [baseline_causations_attr_name]

    def _extract_parts(self, sentence, is_train):
        if is_train:
            return []
        else:
            return getattr(sentence, self.consumed_attributes[0])

    def _make_evaluator(self):
        return IAAEvaluator(False, False, FLAGS.combiner_print_test_instances,
                            True, True)
