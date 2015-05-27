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
    
    def test(self, parts):
        for possible_causation in parts:
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
