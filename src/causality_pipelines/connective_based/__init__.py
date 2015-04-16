from pipeline.models import ClassifierPart

class PossibleCausation(object):
    def __init__(self, matching_pattern, connective_tokens,
                 true_causation_instance):
        # There must be at least 1 connective token, or it's not a valid
        # potential instance anyway.
        self.sentence = connective_tokens[0].parent_sentence
        self.matching_pattern = matching_pattern
        self.connective_tokens = connective_tokens
        self.true_causation_instance = true_causation_instance
        # TODO: Add spans of plausible ranges for argument spans
