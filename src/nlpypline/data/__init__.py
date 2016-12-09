'''
Define basic NLP datatypes
'''

from bidict import bidict
from gflags import DEFINE_bool, FLAGS, DuplicateFlagError
import logging
from nltk.tree import ImmutableParentedTree, Tree
import numpy as np
import re
from scipy.sparse import lil_matrix, csr_matrix, csgraph

from nlpypline.util import Enum, merge_dicts, listify
from nlpypline.util.nltk import (collins_find_heads, nltk_tree_to_graph,
                                 is_parent_of_leaf)
from nlpypline.util.scipy import bfs_shortest_path_costs
from nlpypline.util.streams import (
    CharacterTrackingStreamWrapper, eat_whitespace, is_at_eof,
    peek_and_revert_unless, read_stream_until)


try:
    DEFINE_bool('use_constituency_parse', False,
                'Whether to build constituency parse trees from the provided'
                ' constituency parse string when constructing'
                ' StanfordParsedSentences. Setting to false makes reading in'
                ' data more efficient.')
except DuplicateFlagError:
    pass


class Document(object):
    # TODO: there are probably a lot of other things we should offer here.
    # Starting with the ability to recover the text of the document...
    def __init__(self, filename):
        self.filename = filename

    def __repr__(self):
        return '<%s: %s>' % (self.__class__.__name__, self.filename)


class SentencesDocument(Document):
    def __init__(self, filename, sentences):
        super(SentencesDocument, self).__init__(filename)
        self.sentences = sentences

    def __iter__(self):
        return iter(self.sentences)

    def __getitem__(self, index):
        return self.sentences[index]


class Annotation(object):
    def __init__(self, sentence_offset, offsets, text, annot_id=''):
        ''' offsets is a tuple or list of (start, end) tuples. '''
        self.id = annot_id
        if isinstance(offsets[0], int):
            offsets = (offsets,)
        # Sort because indices could be out of order (particularly in brat)
        self.offsets = sorted(
            [(start_offset - sentence_offset, end_offset - sentence_offset)
             for (start_offset, end_offset) in offsets])
        self.text = text

    def starts_before(self, other):
        return self.offsets[0][0] < other.offsets[0][0]


class Token(object):
    NOUN_TAGS = ["NN", "NP", "NNS", "NNP", "NNPS", "PRP", "WP", "WDT"]
    # TODO:should MD be included below?
    VERB_TAGS = ["VB", "VBD", "VBG", "VBN", "VBP", "VBZ", "MD"]
    ADVERB_TAGS = ["RB", "RBR", "RBS", "WRB"]
    ADJECTIVE_TAGS = ["JJ", "JJR", "JJS"]
    DET_TAGS = ["DT", "EX", "PDT"]
    PUNCT_TAGS = [".", ",", ":", "``", "''", "-LRB-", "-RRB-", "-LCB-", "-RCB-",
                  "-LSB-", "-RSB-"]
    POS_GENERAL = {} # created for real below
    ALL_POS_TAGS = [] # created for real below

    def __init__(self, index, parent_sentence, original_text, pos, lemma,
                 start_offset=None, end_offset=None, is_absent=False,
                 copy_of=None):
        self.index = index
        self.parent_sentence = parent_sentence
        self.original_text = original_text
        self.lowered_text = original_text.lower() # TODO: remove?
        self.pos = pos
        self.lemma = lemma
        self.start_offset = start_offset
        self.end_offset = end_offset
        self.is_absent = is_absent
        self.copy_of = copy_of

    def is_root(self):
        return self.pos == 'ROOT'

    def get_gen_pos(self):
        return Token.POS_GENERAL.get(self.pos, self.pos)

    def __unicode__(self):
        return u"Token(%s_%d/%s [%s:%s])" % (
            self.original_text, self.index, self.pos, self.start_offset,
            self.end_offset)

    def __repr__(self, *args, **kwargs):
        return self.__unicode__().encode('utf-8')


Token.POS_GENERAL = merge_dicts(
    [{tag: 'NN' for tag in Token.NOUN_TAGS},
     {tag: 'VB' for tag in Token.VERB_TAGS},
     {tag: 'JJ' for tag in Token.ADJECTIVE_TAGS},
     {tag: 'RB' for tag in Token.ADVERB_TAGS}])

Token.ALL_POS_TAGS = (Token.NOUN_TAGS + Token.VERB_TAGS + Token.ADVERB_TAGS +
                      Token.ADJECTIVE_TAGS + Token.PUNCT_TAGS + ["IN"])


class DependencyPath(list):
    def __init__(self, path_start, *args, **kwargs):
        super(DependencyPath, self).__init__(*args, **kwargs)
        self.start = path_start

    def __str__(self):
        if not self:
            return ''

        last_node = self.start
        dep_names = []
        for source, target, dep_name in self:
            if source is last_node:
                last_node = target
            else:
                last_node = source
                dep_name += "'"
            dep_names.append(dep_name)
        return ' '.join(dep_names)


class DependencyPathError(ValueError):
    def __init__(self, source, target):
        self.source = source
        self.target = target
        super(DependencyPathError, self).__init__(
            '%s is not reachable from %s' % (target, source))


class StanfordParsedSentence(object):
    PTB_ESCAPE_MAP = {'*': '\\*', '. . .': '...', '(': '-LRB-', ')': '-RRB-',
                      '{': '-LCB-', '}': '-RCB-', '[': '-LSB-', ']': '-RSB-'}
    PTB_UNESCAPE_MAP = {} # filled in later from PTB_ESCAPE_MAP below
    # TODO: Should we be allowing the parser to PTB-escape more things?
    PERIOD_SUBSTITUTES = '.:'
    SUBJECT_EDGE_LABELS = ['nsubj', 'csubj', 'nsubjpass', 'csubjpass']
    INCOMING_CLAUSE_EDGES = ['ccomp', 'xcomp', 'csubj', 'csubjpass', 'advcl',
                             'acl', 'acl:relcl'] # TODO: allow conj/parataxis?
    EDGE_REGEX = re.compile(
        "([A-Za-z_\\-/\\.':]+)\\((.+)-(\\d+)('*), (.+)-(\\d+)('*)\\)")
    DEPTH_EXCLUDED_EDGE_LABELS = ['ref']

    def __init__(self, tokenized_text, tagged_lemmas, penn_tree, edges,
                 document_text):
        '''
        `tokenized_text` and `tagged_lemmas` are the token and lemma strings
         from the parser.
         `edges` is a list of edge strings from the parser.
         `document_text` is an instance of
         `util.streams.CharacterTrackingStreamWrapper`. (Built-in stream types
         will *not* work.)
        '''
        # TODO: move much of the initialization functionality, particularly
        # aligning tokens to text, into the reader class.
        self.tokens = []
        self.edge_labels = {} # maps (n1_index, n2_index) tuples to labels
        try:
            self.source_file_path = document_text.name
        except AttributeError:
            self.source_file_path = None

        # Declare a few variables that will be overwritten later, just so that
        # it's easy to tell what's in an instance of this class.
        self.edge_graph = csr_matrix((0, 0), dtype='float')
        self.document_char_offset = 0
        self.original_text = ''
        self.__depths = np.array([])
        self.path_predecessors = np.array([[]])
        self.path_costs = np.array([[]])

        token_strings, tag_strings = self.__get_token_strings(tokenized_text,
                                                              tagged_lemmas)

        copy_node_indices = self.__create_tokens(token_strings, tag_strings)
        self.__align_tokens_to_text(document_text)
        self.__create_edges(edges, copy_node_indices)

        if FLAGS.use_constituency_parse:
            self.constituency_tree = ImmutableParentedTree.fromstring(penn_tree)
            self.constituency_graph = nltk_tree_to_graph(self.constituency_tree)
            self.constituent_heads = collins_find_heads(self.constituency_tree)
        else:
            self.constituency_tree = None
            self.constituency_graph = None
            self.constituent_heads = None

    @staticmethod
    def unescape_token_text(token_text):
        token_text = token_text.replace(u'\xa0', ' ')
        return StanfordParsedSentence.PTB_UNESCAPE_MAP.get(token_text,
                                                           token_text)

    @staticmethod
    def escape_token_text(token_text):
        token_text = token_text.replace(' ', u'\xa0')
        return StanfordParsedSentence.PTB_ESCAPE_MAP.get(token_text, token_text)

    @staticmethod
    def get_text_for_tokens(annotation_tokens):
        try:
            return ' '.join([token.original_text
                             for token in annotation_tokens])
        except TypeError: # Happens if None is passed
            return ''

    def get_depth(self, token):
        return self.__depths[token.index]

    def _token_is_preferred_for_head_to(self, old_token, new_token):
        # If the depths are equal, prefer verbs/copulas over nouns, and
        # nouns over others. This helps to get the correct heads for
        # fragmented spans, such as spans that consist of an xcomp and its
        # subject, as well as a few other edge cases.
        if self.is_clause_head(new_token):
            return False
        elif self.is_clause_head(old_token):
            return True
        elif new_token.pos in Token.NOUN_TAGS:
            return False
        elif old_token.pos in Token.NOUN_TAGS:
            return True
        else:
            return False

    def get_head(self, tokens):
        # TODO: Update to match SEMAFOR's heuristic algorithm?
        min_depth = np.inf
        head = None
        for token in tokens:
            depth = self.get_depth(token)
            parent_of_current_head = head in self.get_children(token, '*')
            child_of_current_head = (head is not None and
                                     token in self.get_children(head, '*'))
            if ((depth < min_depth or parent_of_current_head)
                and not child_of_current_head):
                head = token
                min_depth = depth
            elif (depth == min_depth and
                  head is not None and
                  self._token_is_preferred_for_head_to(token, head)):
                logging.debug(
                    u"Preferring %s over %s as head of '%s' in '%s'" %
                    (token, head,
                     u' '.join([t.original_text for t in tokens]),
                     tokens[0].parent_sentence.original_text))
                head = token
                min_depth = depth

        if head is None:
            logging.warn('Returning null head for tokens %s'
                         % tokens);
        return head

    def count_words_between(self, token1, token2):
        ''' Counts words between tokens based purely on the token IDs,
            discounting punctuation tokens. '''
        # assert (self.tokens[token1.index] == token1 and
        #        self.tokens[token2.index] == token2), "Tokens not in sentence"
        switch = token1.index > token2.index
        if switch:
            token1, token2 = token2, token1
        words_between = -1
        for token in self.tokens[token1.index : token2.index + 1]:
            if token.pos[0].isalnum():
                words_between += 1
        # return -words_between if switch else words_between
        return words_between

    def get_most_direct_parent(self, token):
        '''
        Returns a tuple (e, p), p is the parent of the given token along the
        shortest path to root, and e is the label of the edge from p to token.
        '''
        # We can't use self.path_predecessors because it was computed in an
        # essentially undirected fashion. Instead, we find all parents, and
        # select the one whose directed depth is lowest (i.e., with the shortest
        # directed path to root).
        incoming = self.edge_graph[:, token.index]
        nonzero = incoming.nonzero()[0]
        if not nonzero.any():
            return (None, None)

        min_depth = np.inf
        for edge_start_index in nonzero:
            next_depth = self.__depths[edge_start_index]
            if next_depth < min_depth:
                min_depth = next_depth
                parent_index = edge_start_index
        edge_label = self.edge_labels[(parent_index, token.index)]
        return (edge_label, self.tokens[parent_index])

    def get_children(self, token, edge_type=None):
        '''
        If `edge_type` is given, returns a list of children of token related by
        an edge with label edge_type. Otherwise, returns a list of
        (edge_label, child_token) tuples.

        `edge_type` may be a single type or a list of types. The special value
        '*' indicates that all children should be returned, without edge labels.
        '''
        # Grab the sparse column of the edge matrix with the edges of this
        # token. Iterate over the edge end indices therein.
        if edge_type:
            if edge_type == '*':
                return [self.tokens[edge_end_index] for edge_end_index
                        in self.edge_graph[token.index].indices]
            else:
                edge_type = listify(edge_type)
                return [self.tokens[edge_end_index] for edge_end_index
                        in self.edge_graph[token.index].indices
                        if (self.edge_labels[(token.index, edge_end_index)]
                            in edge_type)]
        else:
            return [(self.edge_labels[(token.index, edge_end_index)],
                     self.tokens[edge_end_index])
                    for edge_end_index in self.edge_graph[token.index].indices]

    def is_copula_head(self, token):
        # Grab the sparse column of the edge matrix with the edges of this
        # token, and check the labels on each non-zero edge.
        for edge_end_index in self.edge_graph[token.index].indices:
            # A copula edge to a child also indicates a clause.
            if self.edge_labels[(token.index, edge_end_index)] == 'cop':
                return True

    def is_clause_head(self, token):
        if token.pos == 'ROOT':
            return False
        try:
            Token.VERB_TAGS.index(token.pos)
            if token.pos != 'MD': # Modals, though verbs, aren't clause heads
                return True
        except ValueError: # this POS wasn't in the list
            if self.is_copula_head(token):
                return True

            incoming = self.edge_graph[:, token.index]
            for edge_start_index in incoming.nonzero()[0]:
                # An incoming clause edge also indicates a clause.
                if (self.edge_labels[(edge_start_index, token.index)]
                    in self.INCOMING_CLAUSE_EDGES):
                    return True

        return False

    def extract_dependency_path(self, source, target, include_conj=True):
        edges = []
        while target is not source:
            predecessor_index = self.path_predecessors[source.index,
                                                         target.index]
            if predecessor_index == -9999:
                raise DependencyPathError(source, target)
            predecessor = self.tokens[predecessor_index]

            try:
                # Normal case: the predecessor is the source of the edge.
                label = self.edge_labels[(predecessor_index, target.index)]
                start, end = predecessor, target
            except KeyError:
                # Back edge case: the predecessor is the target of the edge.
                label = self.edge_labels[(target.index, predecessor_index)]
                start, end = target, predecessor
            if label != 'conj' or include_conj:
                edges.append((start, end, label))
            target = predecessor
        return DependencyPath(source, reversed(edges))

    def get_closest_of_tokens(self, source, possible_targets, use_tree=True):
        '''
        Finds the token among possible_targets closest to source. If use_tree
        is True, distance is determined by distance in the parse tree;
        otherwise, distance is simple lexical distance (which may be negative).
        Returns the token, along with its distance. If none of the possible
        targets is reachable, returns (None, np.inf).
        '''
        if not possible_targets:
            raise ValueError("Can't find closest of 0 tokens")

        min_distance = np.inf
        for target in possible_targets:
            if use_tree:
                next_distance = self.path_costs[source.index, target.index]
            else:
                next_distance = source.index - target.index
            if next_distance < min_distance:
                closest = target
                min_distance = next_distance
        if min_distance == np.inf: # source or all targets aren't in tree
            closest = None

        return closest, min_distance

    def get_constituency_node_for_tokens(self, tokens):
        # Token indices include ROOT, so we subtract 1 to get indices that will
        # match NLTK's leaf indices.
        indices = [token.index - 1 for token in tokens]
        try:
            treeposition = self.constituency_tree.treeposition_spanning_leaves(
                min(indices), max(indices) + 1) # +1 b/c of Python-style ranges
        except AttributeError: # self.constituency_tree is None
            if not FLAGS.use_constituency_parse:
                raise ValueError('Constituency parses not in use')
            else:
                raise

        node = self.constituency_tree[treeposition]
        if not isinstance(node, Tree): # We got a treeposition of a leaf string
            node = self.constituency_tree[treeposition[:-1]]
        return node

    def get_token_for_constituency_node(self, node):
        if not is_parent_of_leaf(node):
            raise ValueError("Node is not a parent of a leaf: %s" % node)
        node_leaf = node[0]
        for i, leaf in enumerate(node.root().leaves()):
            if leaf is node_leaf: # identity, not equality
                return self.tokens[i]
        if not FLAGS.use_constituency_parse:
            raise ValueError('Constituency parses not in use')
        else:
            raise ValueError("Somehow you passed a node whose leaf isn't under"
                             " its root. Wow.")

    DOMINATION_DIRECTION = Enum(['Dominates', 'DominatedBy', 'Independent'])
    def get_domination_relation(self, token1, token2):
        # TODO: do we need to worry about conj's here?
        path = self.extract_dependency_path(token1, token2, True)
        last_node = path.start
        all_forward = True
        all_backward = True
        for source, target, _dep_name in path:
            if source is last_node: # forward edge
                all_backward = False
                if not all_forward:
                    break
                last_node = target
            else: # back edge
                all_forward = False
                if not all_backward:
                    break
                last_node = source
        if all_forward:
            return self.DOMINATION_DIRECTION.Dominates
        elif all_backward:
            return self.DOMINATION_DIRECTION.DominatedBy
        else:
            return self.DOMINATION_DIRECTION.Independent

    @staticmethod
    def is_contiguous(tokens):
        last_index = tokens[0].index
        for token in tokens[1:]:
            if token.pos in Token.PUNCT_TAGS or token.index == last_index + 1:
                last_index = token.index
            else:
                return False

        return True

    ###########################################
    # Private initialization support functions
    ###########################################

    @staticmethod
    def __get_token_strings(tokenized_text, tagged_lemmas):
        '''
        This is basically a wrapper for the string split function, which also
        combines adjacent tokens if there are spaces within tokens. This is
        detected by looking for a lack of a '/' in the tagged lemma.
        '''
        token_strings = tokenized_text.split(' ')
        lemma_strings = tagged_lemmas.split(' ')
        assert len(token_strings) == len(lemma_strings), (
            "Tokens do not match tags")

        if all('/' in lemma for lemma in lemma_strings):
            return token_strings, lemma_strings

        final_token_strings = []
        final_lemma_strings = []
        tokens_to_accumulate = []
        lemmas_to_accumulate = []
        for token, lemma in zip(token_strings, lemma_strings):
            tokens_to_accumulate.append(token)
            lemmas_to_accumulate.append(lemma)
            if '/' in lemma:
                final_token_strings.append(' '.join(tokens_to_accumulate))
                final_lemma_strings.append(' '.join(lemmas_to_accumulate))
                tokens_to_accumulate = []
                lemmas_to_accumulate = []
        return final_token_strings, final_lemma_strings

    def __create_tokens(self, token_strings, tag_strings):
        # We need one more node than we have token strings (for root).
        copy_node_indices = [None for _ in range(len(token_strings) + 1)]
        root = self.__add_new_token('', 'ROOT', 'ROOT')
        copy_node_indices[0] = [root.index]

        for i, (token_str, tag_str) in (
                enumerate(zip(token_strings, tag_strings))):
            # Can't use str.partition because there may be a '/' in the token.
            slash_index = tag_str.rindex('/')
            lemma = tag_str[:slash_index]
            pos = tag_str[slash_index + 1:]
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
        self.document_char_offset = document_text.character_position

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
                    # print "Found duplicated token:", (
                    #    token.original_text.encode('utf-8'))
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
                    token.start_offset = (document_text.character_position
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
                search_start = document_text.character_position
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
                    (u'Could not find token "%s" starting at position %d '
                     '(accumulated: %s)') % (
                    original, search_start, self.original_text)).encode('utf-8')

                if original[-1] == '.':
                    # If it ends in a period, and the next character in the
                    # stream is a period, it's a duplicated period. Advance
                    # over the period and append it to the accumulated text.
                    _, is_period = peek_and_revert_unless(
                        document_text, lambda char: char == '.')
                    if is_period:
                        self.original_text += '.'
                token.end_offset = (document_text.character_position
                                    - self.document_char_offset)
                token.start_offset = token.end_offset - len(original)

            '''
            if not token.is_absent:
                print "Annotated token span: ", token.start_offset, ",", \
                    token.end_offset, 'for', \
                    token.original_text.encode('utf-8') + '. Annotated text:',\
                    (self.original_text[token.start_offset:token.end_offset]
                    ).encode('utf-8')
            '''

    def __make_token_copy(self, token_index, copy_num, copy_node_indices):
        copies = copy_node_indices[token_index]
        token = self.tokens[token_index]
        while copy_num >= len(copies):
            self.__add_new_token(token.original_text, token.pos, token.lemma,
                                 token.start_offset, token.end_offset,
                                 token.is_absent, token)
            copies.append(len(self.tokens) - 1)

    def __create_edges(self, edges, copy_node_indices):
        edge_lines = [line for line in edges if line] # skip blanks
        matches = [StanfordParsedSentence.EDGE_REGEX.match(edge_line)
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
        self.edge_graph = lil_matrix((num_nodes, num_nodes), dtype='float')
        graph_excluded_edges = [] # edges that shouldn't be used for graph algs
        for match_result in matches:
            (relation, _arg1_lemma, arg1_index, arg1_copy, _arg2_lemma,
             arg2_index, arg2_copy) = match_result.group(*range(1,8))
            arg1_index = int(arg1_index)
            arg2_index = int(arg2_index)

            token_1_idx = copy_node_indices[arg1_index][len(arg1_copy)]
            token_2_idx = copy_node_indices[arg2_index][len(arg2_copy)]
            # TODO: What should we do about the cases where there are
            # multiple labels for the same edge? (e.g., conj and ccomp)
            self.edge_labels[(token_1_idx, token_2_idx)] = relation
            if relation in self.DEPTH_EXCLUDED_EDGE_LABELS:
                graph_excluded_edges.append((token_1_idx, token_2_idx))
            else:
                self.edge_graph[token_1_idx, token_2_idx] = 1.0
        self._initialize_graph(graph_excluded_edges)

    def _initialize_graph(self, graph_excluded_edges):
        # Convert to CSR for shortest path (which would do it anyway) and to
        # make self.get_children() below work.
        self.edge_graph = self.edge_graph.tocsr()
        self.__depths = bfs_shortest_path_costs(self.edge_graph, 0)

        '''
        For the undirected shortest paths we save, we'll want to:
         a) prefer xcomp-> __ ->nsubj paths over nsubj-> __ <-nsubj and
            nsubj<- __ ->xcomp paths.
         b) disprefer paths that rely on expletives and acls.
         c) treat the graph as undirected, EXCEPT for edges where we already
            have a reverse edge, in which case that edge's weight should be
            left alone.
        We adjust the graph accordingly.
        '''
        # Adjust edge weights to make better paths preferred.
        for edge, label in self.edge_labels.iteritems():
            if label == 'xcomp':
                self.edge_graph[edge] = 0.98
                edge_end_token = self.tokens[edge[1]]
                subj_children = self.get_children(
                    edge_end_token, self.SUBJECT_EDGE_LABELS)
                for child in subj_children:
                    self.edge_graph[edge[1], child.index] = 0.985
            elif label == 'expl' or label.startswith('acl'):
                self.edge_graph[edge] = 1.01

        # Create duplicate edges to simulate undirectedness, EXCEPT where we
        # already have an edge in the opposite direction. For this we use a
        # copy of the graph, since we don't actually want to pollute edge_graph
        # with the reverse arcs.
        pseudo_unweighted_graph = self.edge_graph.tolil()

        nonzero = set([(i, j) for (i, j) in zip(
            *pseudo_unweighted_graph.nonzero())])
        for (i, j) in nonzero:
            if (j, i) not in nonzero:
                pseudo_unweighted_graph[j, i] = pseudo_unweighted_graph[i, j]

        self.path_costs, self.path_predecessors = csgraph.shortest_path(
            pseudo_unweighted_graph, return_predecessors=True, directed=True)

        # Add in edges that we didn't want to use for distances/shortest path,
        # ignoring all the changes made to undirected_graph.
        # (Originally we were converting to LIL for efficiency, but it turned
        # out to hurt performance more than it helped.)
        # TODO: Should we convert if there were excluded edges?
        # self.edge_graph = self.edge_graph.tolil()
        for start, end in graph_excluded_edges:
            self.edge_graph[start, end] = 1.0
        self.edge_graph = self.edge_graph.tocsr()

    def __unicode__(self):
        parse_lines = [u'%s(%s-%d, %s-%d)'
                       % (label, self.tokens[edge[0]].lemma, edge[0],
                          self.tokens[edge[1]].lemma, edge[1])
                       for edge, label in sorted(self.edge_labels.iteritems())]
        return u'%s\n\n%s' % (self.original_text, u'\n'.join(parse_lines))

    def __str__(self):
        return self.__unicode__().encode('utf-8')

StanfordParsedSentence.PTB_UNESCAPE_MAP = {
    v: k for k, v in StanfordParsedSentence.PTB_ESCAPE_MAP.items()
}
