from collections import defaultdict
from gflags import FLAGS

from causality_pipelines import IAAEvaluator
from iaa import stringify_connective
from pipeline import Stage
from pipeline.models import Model
from pipeline.models.structured import StructuredModel, StructuredDecoder


class MostFreqSenseFilterModel(StructuredModel, StructuredDecoder):
    def __init__(self):
        super(MostFreqSenseFilterModel, self).__init__(self)
        self.reset()

    def _make_parts(self, instance, is_train):
        return instance.possible_causations
    
    def _train_structured(self, instances, parts_by_instance):
        for instance_pcs in parts_by_instance:
            for possible_causation in instance_pcs:
                label = bool(possible_causation.true_causation_instance)
                connective = stringify_connective(possible_causation)
                self.frequencies_dict[connective][label] += 1
        
        for key in self.frequencies_dict:
            negative, positive = self.frequencies_dict[key]
            self.frequencies_dict[key] = positive > negative

    def _score_parts(self, sentence, possible_causations):
        return [self.frequencies_dict[stringify_connective(pc)]
                for pc in possible_causations]

    def decode(self, sentence, possible_causations, scores):
        return [pc for pc, score in zip(possible_causations, scores) if score]

    def reset(self):
        self.frequencies_dict = defaultdict(lambda: [0, 0])


class MostFreqSenseFilterStage(Stage):
    def __init__(self, name):
        super(MostFreqSenseFilterStage, self).__init__(
            name=name, model=MostFreqSenseFilterModel())

    consumed_attributes = ['possible_causations']

    def _label_instance(self, document, sentence, predicted_causations):
        sentence.causation_instances = predicted_causations

    def _make_evaluator(self):
        return IAAEvaluator(False, False,
                            FLAGS.causality_cc_print_test_instances, True, True)
