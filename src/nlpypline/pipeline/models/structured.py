from gflags import DEFINE_bool, FLAGS, DuplicateFlagError
from itertools import chain # , izip_longest
import logging
import numpy as np
import pycrfsuite
import time
from types import MethodType

from nlpypline.pipeline.models import Model, MultiplyFeaturizedModel
from nlpypline.pipeline.featurization import DictOnlyFeaturizer, Featurizer

try:
    DEFINE_bool('pycrfsuite_verbose', False,
                'Verbose logging output from python-crfsuite trainer')
except DuplicateFlagError as e:
    logging.warn('Ignoring flag redefinitions; assuming module reload')


class StructuredModel(Model):
    '''
    In a structured model, every instance is divided up into "parts." Those
    parts are treated as the thing to be scored by the model. Thus, this class
    overrides the default train and test methods to extract parts first, and
    then call the normal test/train on the parts rather than the instances.
    (Thus, it's often a good idea for the parts to store pointers to the
    original instances for use in featurization, as the feature extractors won't
    get a copy of the original instance on the side.)

    A StructuredModel also has a StructuredDecoder, which is used to decode the
    scored parts into a coherent labeling for the instance.
    '''

    def __init__(self, decoder, *args, **kwargs):
        ''' decoder is some StructuredDecoder object. '''
        self.decoder = decoder
        super(StructuredModel, self).__init__(*args, **kwargs)

    def _train_model(self, instances):
        self.reset() # Reset state in case we've been previously trained.
        parts_by_instance = [self._make_parts(instance, True)
                             for instance in instances]
        return self._train_structured(instances, parts_by_instance)

    # TODO: should structured models have Trainers, like in scikit-learn?
    # That would allow, e.g., training the same model with 2 different methods.
    def _train_structured(self, instances, parts_by_instance):
        raise NotImplementedError

    def test(self, instances):
        # In structured models, we often want to incrementally write out results
        # one instance at a time, so yield results rather than creating a list.
        for instance in instances:
            instance_parts = self._make_parts(instance, False)
            part_scores = self._score_parts(instance, instance_parts)
            yield self.decoder.decode(instance, instance_parts, part_scores)

    def _make_parts(self, instance, is_train):
        raise NotImplementedError

    def _score_parts(self, instance, instance_parts):
        raise NotImplementedError


class FeaturizedStructuredModel(StructuredModel, MultiplyFeaturizedModel):
    '''
    If there is more than one part type, self.featurizers should contain one
    featurizer per part type (since each part type will be featurized
    separately).
    '''
    def __init__(self, decoder, part_types, selected_features=None,
                 part_filters=None, model_path=None, save_featurized=False,
                 *args, **kwargs):
        """
        decoder is some StructuredDecoder object.
        part_types is a list of types of part that will need to be featurized
            separately (e.g., node and edge parts). These should be the actual
            Python types of the parts returned by _make_parts (or supertypes
            thereof).
        selected_features is a list of names of features to extract.
            Names may be conjoined by FLAGS.conjoined_feature_sep. If there is
            more than one part type, then selected_features should be a list of
            lists of feature names, one list per part type.
        part_filters is a list filter functions, corresponding to part_types,
            that take an part and return True iff it should be featurized. Parts
            that are filtered out will be featurized as all zeros.
        model_path is a path to a model to load. Either model_path or
            selected_features must be specified.
        save_featurized indicates whether to store features and labels
            properties after featurization. Useful for debugging/development.
        """
        if not part_filters:
            part_filters = [None] * len(part_types)
        else:
            assert len(part_filters) == len(part_types)
        self._part_filters = part_filters
        self.part_types = part_types

        super(FeaturizedStructuredModel, self).__init__(
            decoder=decoder, selected_features=selected_features,
            model_path=model_path, save_featurized=save_featurized,
            *args, **kwargs)

    def _make_featurizer(self, extractors, featurizer_params, featurizer_index):
        return Featurizer(extractors, featurizer_params, self.save_featurized,
                          self._part_filters[featurizer_index])

    def _score_parts(self, instance, instance_parts):
        featurized_parts_by_type = []
        for part_type, featurizer in zip(self.part_types, self.featurizers):
            relevant_parts = [part for part in instance_parts
                              if isinstance(part, part_type)]
            featurized = featurizer.featurize(relevant_parts)
            featurized_parts_by_type.append(featurized)

        return self._score_featurized_parts(instance, featurized_parts_by_type)

    def _train_structured(self, instances, parts_by_instance):
        # TODO: Does this need to be broken up by instance?
        all_parts = list(chain.from_iterable(parts_by_instance))
        featurized_with_labels_by_type = []

        for part_type, featurizer in zip(self.part_types, self.featurizers):
            relevant_parts = [part for part in all_parts
                              if isinstance(part, part_type)]
            logging.info("Registering features for %ss..." % part_type.__name__)
            featurizer.register_features_from_instances(relevant_parts)
            logging.info('Done registering features.')

            featurized = featurizer.featurize(relevant_parts)
            part_labels = self._get_gold_labels(part_type, relevant_parts)
            featurized_with_labels_by_type.append((featurized, part_labels))

        self._train_featurized_structured(featurized_with_labels_by_type)

    def _score_featurized_parts(self, instance, featurized_parts_by_type):
        raise NotImplementedError

    def _train_featurized_structured(self, featurized_with_labels_by_type):
        raise NotImplementedError

    def _get_gold_labels(self, part_type, parts):
        raise NotImplementedError


class StructuredDecoder(object):
    def __init__(self, save_scored=False):
        self.save_scored = save_scored

    def decode(self, instance, instance_parts, scores):
        raise NotImplementedError


# TODO: move code below to separate sequences submodule?
class SequenceScores(object):
    def __init__(self, node_scores, transition_weights):
        self.node_scores = node_scores
        self.transition_weights = transition_weights


class Semiring(object):
    def __init__(self, np_sum, np_multiply, additive_identity,
                 multiplicative_identity, np_arg_sum=None):
        self.sum = np_sum
        self.multiply = np_multiply
        self.additive_identity = additive_identity
        self.multiplicative_identity = multiplicative_identity
        self.arg_sum = np_arg_sum

# Common semirings
Semiring.PLUS_MULTIPLY = Semiring(np.sum, np.multiply, 0, 1) # count/probability
Semiring.MAX_MULTIPLY = Semiring(np.max, np.multiply, -np.inf, 1, np.argmax)
Semiring.MAX_PLUS = Semiring(np.max, np.add, -np.inf, 0, np.argmax)


class ViterbiDecoder(StructuredDecoder):
    def __init__(self, possible_states=None, semiring=Semiring.MAX_MULTIPLY):
        self.possible_states = possible_states
        self.semiring = semiring

    # TODO: allow converting to log space to deal with numerical stability
    # (maybe by just offering a log-based semiring?)
    def run_viterbi(self, node_scores, transition_weights,
                    return_best_path=True):
        '''
        node_scores is a numpy array of scores for individual trellis nodes
            (size: num_states x num_sequence_items). Any start probabilities/
            weights are assumed to be folded into the first column of scores.
        transition_weights is one of:
          - a num_states x num_states array of scores for transitioning between
            states.
          - a (num_sequence_items-1) x num_states x num_states array of scores
            for transitioning between states for particular sequence items.
        if return_best_path is True, then instead of just returning the best
            score, the function will return (summed_score, best_state_path).
            (The semiring must have arg_sum defined for this.)
        '''
        # TODO: generalize code to higher Markov orders?
        assert self.semiring.arg_sum or not return_best_path, ('Can only return'
            ' best path for semirings with a defined arg_sum operation')

        # Declare arrays and initialize to base case values
        path_scores = np.empty(node_scores.shape)
        path_scores[:, 0] = node_scores[:, 0]
        if return_best_path:
            predecessors = np.empty(node_scores.shape, dtype=np.int32)
            predecessors[:, 0] = np.NaN

        transition_scores_by_item = transition_weights.ndim > 2
        # Recursive case: compute each trellis column based on previous column
        num_columns = node_scores.shape[1]
        for column_index in range(1, num_columns):
            # Find best predecessor state for each state.
            if transition_scores_by_item:
                pred_transition_weights = transition_weights[
                    column_index - 1, :, :] # -1 b/c #transitions = #items - 1
            else:
                pred_transition_weights = transition_weights
            # predecessor_scores will be num_states x num_states.
            # Rows represent start states and columns represent end states for
            # this transition.
            predecessor_scores = self.semiring.multiply(
                pred_transition_weights, path_scores[:, column_index - 1,
                                                     np.newaxis])

            if return_best_path:
                predecessor_indices = self.semiring.arg_sum(
                    predecessor_scores, axis=0) # "sum" over start states
                predecessors[:, column_index] = predecessor_indices
                # This "sum" is really a max or a min -- it just selects one
                # predecessor for each state.
                summed_scores = predecessor_scores[
                    predecessor_indices, range(len(predecessor_scores))]
            else:
                summed_scores = self.semiring.sum(predecessor_scores, axis=0)

            path_scores[:, column_index] = self.semiring.multiply(
                node_scores[:, column_index], summed_scores)

        if return_best_path:
            # Now reconstruct the best sequence from the predecessors matrix.
            best_state_path = np.empty((num_columns,), dtype=np.int32)
            best_final_index = self.semiring.arg_sum(path_scores[:, -1])
            best_state_path[-1] = best_final_index
            summed_score = path_scores[best_final_index, -1]
            for i in reversed(range(1, num_columns)):
                best_state_path[i - 1] = predecessors[best_state_path[i], i]

            if self.possible_states:
                best_state_path = [self.possible_states[i]
                                   for i in best_state_path]

            return summed_score, best_state_path
        else:
            summed_score = self.semiring.sum(path_scores[:, -1])
            return summed_score

    def decode(self, instance, instance_parts, scores):
        # (Rows = states, columns = sequence items.)
        best_score, best_path = self.run_viterbi(
            scores.node_scores, scores.transition_weights, True)

        logging.debug("Viterbi max score: %d", best_score)
        return best_path


class CRFDecoder(StructuredDecoder):
    def decode(self, instance, instance_parts, part_labels_by_type):
        return part_labels_by_type[0]


class CRFModel(FeaturizedStructuredModel):
    class CRFTrainingError(Exception):
        pass

    class CRFPart(object):
        '''
        In a CRF model, the feature extractors will have to operate on a single
        observation from a sequence, with the context of the sequence of
        surrounding observations. This class encapsulates the data a CRF feature
        extractor may need.
        '''
        def __init__(self, observation, sequence, index, instance):
            self.observation = observation
            self.sequence = sequence
            self.index = index
            self.instance = instance

    def __init__(self, selected_features, model_file_path, training_algorithm,
                 training_params, decoder=CRFDecoder(), part_filters=None,
                 load_from_file=False, save_featurized=False, *args, **kwargs):
        self.model_file_path = model_file_path
        if not load_from_file:
            model_file_path = None # to tell the super constructor not to load
        super(CRFModel, self).__init__(
            decoder=decoder, part_types=[CRFModel.CRFPart],
            selected_features=selected_features, part_filters=part_filters,
            model_path=model_file_path, save_featurized=save_featurized,
            *args, **kwargs)
        self.training_algorithm = training_algorithm
        self.training_params = training_params
        self.tagger = None

    # Override featurizer creation to make default featurization produce a dict
    # of feature values rather than a matrix.
    def _make_featurizer(self, extractors, featurizer_params, featurizer_index):
        part_filter = self._part_filters[featurizer_index]
        return DictOnlyFeaturizer(extractors, featurizer_params,
                                  self.save_featurized, part_filter)

    @staticmethod
    def __handle_training_error(trainer, log):
        raise CRFModel.CRFTrainingError('CRF training failed: %s' % log)

    def _train_featurized_structured(self, featurized_with_labels_by_type):
        trainer = pycrfsuite.Trainer(verbose=FLAGS.pycrfsuite_verbose)
        trainer.select(self.training_algorithm)
        trainer.set_params(self.training_params)
        error_handler = MethodType(self.__handle_training_error, trainer)
        trainer.on_prepare_error = error_handler

        # There's only one part type for a CRF.
        featurized_with_labels = featurized_with_labels_by_type[0]
        observation_features, labels = featurized_with_labels
        trainer.append(observation_features, labels)
        '''
        print "Featurized train:"
        for o, l in izip_longest(observation_features, labels):
            print l, 'part:', o
        # '''

        start_time = time.time()
        logging.info("Training CRF model...")
        trainer.train(self.model_file_path)
        elapsed_seconds = time.time() - start_time
        logging.info('CRF model saved to %s (training took %0.2f seconds)'
                     % (self.model_file_path, elapsed_seconds))

    def _make_parts(self, instance, is_train):
        sequence = self._sequence_for_instance(instance, is_train)
        return [self.CRFPart(observation, sequence, i, instance)
                for i, observation in enumerate(sequence)]

    def _load_model(self, filepath):
        self.tagger = pycrfsuite.Tagger()
        self.tagger.open(self.model_file_path)

    def save(self, filepath): # Saving is built into CRFSuite
        pass

    def _post_model_train(self):
        super(CRFModel, self)._post_model_train()
        # Initialize tagger from on-disk file
        self._load_model(self.model_file_path)

    def _score_featurized_parts(self, instance, featurized_parts_by_type):
        featurized_parts = featurized_parts_by_type[0] # only 1 part type
        crf_labels = self.tagger.tag(featurized_parts)
        '''
        print "Featurized test:"
        for o, l in izip_longest(featurized_parts, crf_labels):
            print l, 'part:', o
        # '''
        return [crf_labels]

    def _sequence_for_instance(self, instance, is_train):
        raise NotImplementedError

    def _get_feature_extractor_groups(self):
        return [self.all_feature_extractors]

    # Subclasses should override this class-level variable to include actual
    # feature extractor objects.
    all_feature_extractors = []
