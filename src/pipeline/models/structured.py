import logging
import numpy as np

from pipeline.models import FeaturizedModel


# TODO: separate this from FeaturizedModel and make it a mixin?
class StructuredModel(FeaturizedModel):
    '''
    In a structured model, every instance is divided up into "parts." Those
    parts are treated as the thing to be scored by the model. Thus, this class
    overrides the default train and test methods to extract parts first, and
    then call the normal test/train on the parts rather than the instances.
    (Thus, it's often a good idea for the parts to store pointers to the
    original instances for use in featurization, as the feature extractors won't
    get a copy of the original instance on the side.)
    
    A StructuredModel also has a StructuredDecoder, which is used to decode the scored
    parts into a coherent labeling for the instance.
    '''

    def __init__(self, feature_extractors, decoder, selected_features=None,
                 model_path=None, save_featurized=False):
        """
        feature_extractors is a list of
            `pipeline.feature_extractors.FeatureExtractor` objects. They must be
            set up to extract features from parts, not instances.
        decoder is some StructuredDecoder object.
        selected_features is a list of names of features to extract. Names may
            be combinations of feature names, separated by ':'.
        save_featurized indicates whether to store features and labels
            properties after featurization. Useful for debugging/development.
        """
        super(StructuredModel, self).__init__(
            feature_extractors, selected_features, model_path, save_featurized)
        self.decoder = decoder

    def train(self, instances):
        self.reset() # Reset state in case we've been previously trained.
        parts_by_instance = [self.make_parts(instance)
                             for instance in instances]

        logging.info("Registering features...")
        # The things we'll be featurizing are parts, so that's what we register
        # features for.
        for instance_parts in parts_by_instance:
            self._register_features(instance_parts)
        logging.info("Done registering features.")

        self._featurized_train(parts_by_instance)

    def _featurized_test(self, instances):
        parts_by_instance = [self.make_parts(instance)
                             for instance in instances]
        outputs_by_instance = []
        for instance_parts in parts_by_instance:
            part_scores = self._score_parts(instance_parts)
            outputs_by_instance.append(
                self.decoder.decode(instance_parts, part_scores))
        return outputs_by_instance

    def make_parts(self, tokens):
        raise NotImplementedError

    # Override featurized train method just to rename the argument so it's
    # clear what subclasses should expect to be passed in.
    def _featurized_train(self, parts_by_instance):
        raise NotImplementedError

    def _score_parts(self, instance_parts):
        raise NotImplementedError


class StructuredDecoder(object):
    def decode(self, instance_parts, scores):
        raise NotImplementedError


class ViterbiScores(object):
    def __init__(self, node_scores, transition_weights=None):
        self.node_scores = node_scores
        self.transition_weights = transition_weights


class ViterbiSemiring(object):
    def __init__(self, np_sum, np_arg_sum, np_multiply,
                 additive_identity, multiplicative_identity):
        self.sum = np_sum
        self.arg_sum = np_arg_sum
        self.multiply = np_multiply
        self.additive_identity = additive_identity
        self.multiplicative_identity = multiplicative_identity

# Common semirings
ViterbiSemiring.MAX_MULTIPLY = ViterbiSemiring(np.max, np.argmax, np.multiply,
                                               - np.inf, 1) # TODO: FIX SPACING
ViterbiSemiring.MAX_ADD = ViterbiSemiring(np.max, np.argmax, np.add, -np.inf,
                                          0)


class ViterbiDecoder(StructuredDecoder):
    def __init__(self, possible_states, semiring=ViterbiSemiring.MAX_MULTIPLY):
        self.possible_states = possible_states
        self.semiring = semiring

    @staticmethod
    def run_viterbi(num_states, node_scores, transition_weights,
                    semiring=ViterbiSemiring.MAX_MULTIPLY,
                    return_best_path=True):
        '''
        num_states is the number of possible hidden states.
        node_scores is a numpy array of scores for individual trellis nodes.
        transition_weights is a num_states x num_states array of scores for
            transitioning between states.
        semiring is an object of type ViterbiSemiring.
        if return_best_path is True, then instead of just returning the best
            score, the function will return (summed_score, best_state_path).
            (The semiring must have arg_sum defined for this.)
        '''
        # TODO: generalize code to higher Markov orders?
        assert semiring.arg_sum or not return_best_path, ('Can only return'
            ' best path for semirings with a defined arg_sum')

        # Declare arrays and initialize to base case values
        path_scores = np.empty(node_scores.shape)
        path_scores[:, 0] = node_scores[:, 0]
        if return_best_path:
            predecessors = np.empty(node_scores.shape)
            predecessors[:, 0] = np.NaN

        # Recursive case: compute each trellis column based on previous column
        num_columns = node_scores.shape[1]
        for column_index in range(1, num_columns):
            # Find best predecessor state for each state.
            predecessor_scores = semiring.multiply(
                transition_weights, path_scores[:, column_index - 1])
            if return_best_path:
                predecessor_indices = semiring.arg_sum(predecessor_scores,
                                                       axis=0)
                summed_scores = path_scores[predecessor_indices,
                                            column_index - 1]
            else:
                summed_scores = semiring.sum(predecessor_scores, axis=0)

            path_scores[:, column_index] = semiring.multiply(
                node_scores[:, column_index], summed_scores)

            if return_best_path:
                predecessors[:, column_index] = predecessor_indices

        if return_best_path:
            # Now reconstruct the best sequence from the predecessors matrix.
            best_state_path = np.empty((num_columns,))
            best_final_index = semiring.arg_sum(path_scores[:, -1])
            best_state_path[-1] = best_final_index
            summed_score = path_scores[best_final_index, -1]
            for i in reversed(range(1, num_columns)):
                best_state_path[i - 1] = predecessors[best_state_path[i], i]

            return summed_score, best_state_path
        else:
            summed_score = semiring.sum(path_scores[:, -1])
            return summed_score
    
    def decode(self, instance_parts, scores):
        # (Rows = states, columns = sequence items.)
        num_states = scores.node_scores.shape[0]
        if not scores.transition_weights:
            scores.transition_weights = np.full(
                (num_states, num_states), self.semiring.multiplicative_identity)
        best_score, best_path = self.run_viterbi(
            num_states, scores.node_scores, scores.transition_weights,
            self.semiring, True)

        logging.debug("Viterbi max score: %d", best_score)
        return best_path
