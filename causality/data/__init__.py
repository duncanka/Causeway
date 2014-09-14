'''
Define basic causality datatypes
'''

import numpy as np
import re
from scipy.sparse import lil_matrix, csr_matrix, csgraph
from util import Enum
from util.streams import *
import warnings

class Annotation(object):
    def __init__(self, sentence_offset, offsets, text, annot_id=''):
        ''' offsets is a tuple or list of (start, end) tuples. '''
        self.id = annot_id
        if isinstance(offsets[0], int):
            offsets = (offsets,)
        self.offsets = \
            [(start_offset - sentence_offset, end_offset - sentence_offset)
             for (start_offset, end_offset) in offsets]
        self.text = text

class Token(object):
    def __init__(self, index, parent_sentence, original_text, pos, lemma,
                 start_offset=None, end_offset=None, is_absent=False,
                 copy_of=None):
        self.index = index
        self.parent_sentence = parent_sentence
        self.original_text = original_text
        self.pos = pos
        self.lemma = lemma
        self.start_offset = start_offset
        self.end_offset = end_offset
        self.is_absent = is_absent
        self.copy_of = copy_of

    def is_root(self):
        return self.pos == 'ROOT'

class ParsedSentence(object):
    UNESCAPE_MAP = {'\\*': '*', '...': '. . .'}
    PERIOD_SUBSTITUTES = '.:'
    NOUN_TAGS = ["NN", "NNS", "NNP", "NNPS", "PRP", "WP"]
    # TODO:should MD be included below?
    VERB_TAGS = ["VB", "VBD", "VBG", "VBN", "VBP", "VBZ", "MD"]

    @staticmethod
    def unescape_token_text(token_text):
        return ParsedSentence.UNESCAPE_MAP.get(token_text, token_text)

    def __init__(self, tokenized_text, tagged_lemmas, edges, document_text):
        self.tokens = []
        self.causation_instances = []
        self.edge_labels = {} # maps (n1_id, n2_id) tuples to labels
        try:
            self.source_file_path = document_text.name
        except AttributeError:
            self.source_file_path = None

        # Declare a few variables that will be overwritten later, just so that
        # it's easy to tell what's in an instance of this class.
        self.edge_graph = csr_matrix((0, 0), dtype='bool')
        self.document_char_offset = -1
        self.original_text = ''
        self.__shortest_distances = np.array([[]])

        token_strings = tokenized_text.split(' ')
        tag_strings = tagged_lemmas.split(' ')
        assert len(token_strings) == len(tag_strings), "Tokens do not match tags"

        copy_node_indices = self.__create_tokens(token_strings, tag_strings)
        self.__align_tokens_to_text(document_text)
        self.__create_edges(edges, copy_node_indices)

    def get_depth(self, token):
        return self.__shortest_distances[0][token.index]

    def get_head(self, annotation):
        min_depth = float('inf')
        head = None
        for token in self.tokens[1:]: # skip ROOT
            depth = self.get_depth(token)
            if depth < min_depth:
                for offset_pair in annotation.offsets:
                    if (token.start_offset >= offset_pair[0] and
                        token.end_offset <= offset_pair[1]):
                        # We've found a token that's higher than our existing 
                        # best candidate, and which is contained by one of the
                        # annotation ranges.
                        head = token
                        min_depth = depth
                        break

        if head is None:
            warnings.warn('Returning null head for annotation "%s"'
                          % annotation.text);
        return head

    def add_causation_instance(self, instance):
        self.causation_instances.append(instance)

    def count_words_between(self, token1, token2):
        ''' Counts words between tokens based purely on the token IDs, 
            discounting punctuation tokens. '''
        assert (self.tokens[token1.index] == token1 and 
                self.tokens[token2.index] == token2), "Tokens not in sentence"
        sentence = token1.parent_sentence
        words_between = -1
        for token in self.tokens[token1.index : token2.index + 1]:
            if token.pos[0].isalnum():
                words_between += 1
        return words_between

    def is_clause(self, token):
        if self.VERB_TAGS.index(token.pos) != -1:
            return True
        else:
            # Grab the sparse column of the edge matrix with the edges of this
            # token, and check the labels on each non-zero edge.
            for edge_end_index in self.edge_graph[token.index].indices:
                # A copula edge to a child also indicates a clause.
                if self.edge_labels[(token.index, edge_end_index)] == 'cop':
                    return True

        return False

    def token_texts(self):
        return [t.original_text for t in self.tokens[1:]]


    ''' Private support functions '''

    def __create_tokens(self, token_strings, tag_strings):
        # We need one more node than we have token strings (for root).
        copy_node_indices = [None for _ in range(len(token_strings) + 1)]
        root = self.__add_new_token('', 'ROOT', 'ROOT')
        copy_node_indices[0] = [root.index]

        for i, (token_str, tag_str) in (
                enumerate(zip(token_strings, tag_strings))):
            lemma, _, pos = tag_str.partition('/')
            new_token = self.__add_new_token(
                self.unescape_token_text(token_str), pos, lemma)
            # Detect duplicated tokens.
            if (lemma == '.' and pos == '.'
                    # Previous token is in self.tokens[i], not i-1: root is 0.
                    and self.tokens[i].original_text.endswith('.')):
                new_token.is_absent = True

            copy_node_indices[i + 1] = [new_token.index]
        
        return copy_node_indices

    def __add_new_token(self, *args, **kwargs):
        new_token = Token(len(self.tokens), self, *args, **kwargs)
        self.tokens.append(new_token)
        return new_token

    def __align_tokens_to_text(self, document_text):
        eat_whitespace(document_text)
        self.document_char_offset = document_text.tell()

        # Root has no alignment to source.
        self.tokens[0].start_offset = None
        self.tokens[0].end_offset = None

        non_root_tokens = self.tokens[1:]
        for i, token in enumerate(non_root_tokens):
            # i is one less than the index of the current token in self.tokens,
            # because root.
            original = token.original_text
            if token.is_absent:
                # Handle case of duplicated character, which is the only type of
                # absent token that will have been detected so far.
                prev_token = self.tokens[i]
                if prev_token.original_text.endswith(original):
                    #print "Found duplicated token:", token.original_text
                    token.start_offset = prev_token.end_offset - len(original)
                    token.end_offset = prev_token.end_offset
            elif original == '.' and i == len(non_root_tokens) - 1:
                # End-of-sentence period gets special treatment: the "real" 
                # original text may have been a period substitute or missing.
                # (Other things can get converted to fake end-of-sentence
                # periods to make life easier for the parser.)
                start_pos = document_text.tell()
                eaten_ws = eat_whitespace(document_text, True)
                not_at_eof = not is_at_eof(document_text)
                next_char, next_is_period_sub = peek_and_revert_unless(
                    document_text,
                    lambda char: self.PERIOD_SUBSTITUTES.find(char) != -1)
                if (not_at_eof and next_is_period_sub):
                    # We've moved the stream over the period, so adjust offset.
                    token.start_offset = (document_text.tell()
                                          - self.document_char_offset - 1)
                    token.end_offset = token.start_offset + 1
                    token.original_text = next_char
                    self.original_text += eaten_ws + next_char
                else:
                    # The period is actually not there.
                    token.is_absent = True
                    token.original_text = ''
                    document_text.seek(start_pos)
            else: # Normal case: just read the next token.
                search_start = document_text.tell()
                # Our preprocessing may hallucinate periods onto the ends of
                # abbreviations, particularly "U.S." Deal with them.
                if original[-1] == '.':
                    token_text_to_find = original[:-1]
                else:
                    token_text_to_find = original
                
                text_until_token, found_token = (
                    read_stream_until(document_text, token_text_to_find, True))
                self.original_text += text_until_token
                assert found_token, (
                    ('Could not find token "%s" starting at position %d '
                     '(accumulated: %s)')
                    % (original, search_start, self.original_text))

                if original[-1] == '.':
                    # If it ends in a period, and the next character in the
                    # stream is a period, it's a duplicated period. Advance
                    # over the period and append it to the accumulated text.
                    _, is_period = peek_and_revert_unless(
                        document_text, lambda char: char == '.')
                    if is_period:
                        self.original_text += '.'
                token.end_offset = (document_text.tell() 
                                    - self.document_char_offset)
                token.start_offset = token.end_offset - len(original)
            
            '''
            if not token.is_absent:
                print "Annotated token span: ", token.start_offset, ",", \
                    token.end_offset, 'for', token.original_text + \
                    '. Annotated text:', \
                    self.original_text[token.start_offset:token.end_offset]
            '''
                

    def __make_token_copy(self, token_index, copy_num, copy_node_indices):
        copies = copy_node_indices[token_index]
        token = self.tokens[token_index]
        while copy_num >= len(copies):
            self.__add_new_token(token.original_text, token.pos, token.lemma,
                                 token.start_offset, token.end_offset, 
                                 token.is_absent, token)
            copies.append(len(self.tokens) - 1)

    EDGE_REGEX = re.compile(
        "([A-Za-z_\\-/\\.']+)\\((.+)-(\\d+)('*), (.+)-(\\d+)('*)\\)")

    def __create_edges(self, edges, copy_node_indices):
        edge_lines = [line for line in edges if line] # skip blanks
        matches = [ParsedSentence.EDGE_REGEX.match(edge_line) 
                   for edge_line in edge_lines]

        # First, we need to create tokens for all the copy nodes so that we have
        # the right size matrix for the graph.
        for match_result, edge_line in zip(matches, edge_lines):
            assert match_result, \
                'Improperly constructed edge line: %s' % edge_line
            arg1_index, arg1_copy, arg2_index, arg2_copy = \
                match_result.group(3, 4, 6, 7)
            self.__make_token_copy(int(arg1_index), len(arg1_copy), 
                                   copy_node_indices)
            self.__make_token_copy(int(arg2_index), len(arg2_copy), 
                                   copy_node_indices)

        # Now, we can actually create the matrix and insert all the edges.
        num_nodes = len(self.tokens)
        self.edge_graph = lil_matrix((num_nodes, num_nodes), dtype='bool')
        for match_result in matches:
            (relation, arg1_lemma, arg1_index, arg1_copy, arg2_lemma, 
             arg2_index, arg2_copy) = match_result.group(*range(1,8))
            arg1_index = int(arg1_index)
            arg2_index = int(arg2_index)
           
            token_1_idx = copy_node_indices[arg1_index][len(arg1_copy)]
            token_2_idx = copy_node_indices[arg2_index][len(arg2_copy)]
            self.edge_graph[token_1_idx, token_2_idx] = True
            self.edge_labels[(token_1_idx, token_2_idx)] = relation
        self.edge_graph = self.edge_graph.tocsr()

        self.__shortest_distances = csgraph.shortest_path(
            self.edge_graph, unweighted=True)

class CausationInstance(object):
    Degrees = Enum(['Entail', 'Facilitate', 'Enable', 'Disentail', 'Inhibit', 
                    'Prevent'])
    CausationTypes = Enum(['Consequence', 'Implication', 'Motivation', 
                           'Purpose'])

    def __init__(self, source_sentence, degree=None, causation_type=None,
                 connective=None, cause=None, effect=None, annotation_id=None):
        if degree is None:
            degree = len(self.Degrees)
        if causation_type is None:
            degree = len(self.CausationTypes)

        assert source_sentence is not None
        self.source_sentence = source_sentence
        self.degree = degree
        self.type = causation_type
        self.connective = connective
        self.cause = cause
        self.effect = effect
        self.id = annotation_id

    def get_cause_and_effect_heads(self):
        if self.cause:
            cause = self.source_sentence.get_head(self.cause)
        else:
            cause = None

        if self.effect:
            effect = self.source_sentence.get_head(self.effect)
        else:
            effect = None

        return (cause, effect)
