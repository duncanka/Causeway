"""
Tweaks/tools for NLTK.
"""

from __future__ import absolute_import
from itertools import islice
import logging
from nltk.tree import Tree, ImmutableTree
from scipy.sparse import lil_matrix

# Weird hack: make ImmutableTree objects return themselves on deep copy.
# They are by definition immutable, so this should be safe. (This is exactly
# what deepcopy already does for tuples.)
def deepcopy_immutable_tree(self, memo):
    return self
setattr(ImmutableTree, '__deepcopy__', deepcopy_immutable_tree)

def is_parent_of_leaf(tree):
    return tree and (isinstance(tree[0], str) or isinstance(tree[0], unicode))

def nltk_tree_to_graph(root):
    '''
    Creates a Scipy CSR graph representing the constituency parse tree defined
    by `tree`. Node indices of the new graph correspond to a left-to-right
    preorder walk of the tree, excluding leaf nodes (words). These indices can
    be matched against nltk.tree.Tree.subtrees().
    '''
    subtrees = [t for t in root.subtrees()]
    num_nodes = len(subtrees)
    graph = lil_matrix((num_nodes, num_nodes), dtype=bool)
    def convert(tree, tree_index):
        # TODO: Is there a more straightforward/efficient way to do this?
        num_processed = 0
        for subtree in tree:
            if isinstance(subtree, Tree):
                num_processed += 1
                subtree_index = tree_index + num_processed
                graph[tree_index, subtree_index] = True
                num_processed += convert(subtree, subtree_index)
        return num_processed
    convert(root, 0)
    return graph.tocsr()

def subtree_at_index(tree, index):
    return next(islice(tree.subtrees(), index, index + 1))

def index_of_subtree(subtree, root=None):
    '''
    If root is not provided, subtree must be parented so that the root can be
    recovered.
    '''
    if root is None:
        root = subtree.root()
    for i, root_subtree in enumerate(root.subtrees()):
        if root_subtree is subtree:
            return i
    raise ValueError(
        'Somehow you passed a subtree that was not under its root. Good job.')

#########################
# Head finding
#########################

# Based on https://berkeley-coreference-analyser.googlecode.com/git/nlp_util/head_finder.py
# (revision e44e5359ba81).
# See http://www.cs.columbia.edu/~mcollins/papers/heads.
#
# NOTE: This implementation assumes the use of nltk.tree.ParentedTree (or a
# subclass of it). The only thing that relies on the tree being parented is the
# warning for unknown label types.
#
# 'TO' has been removed from VP and S, as per the Magerman table.

collins_mapping_table = {
  'ADJP': ('right', ['NNS', 'QP', 'NN', '$', 'ADVP', 'JJ', 'VBN', 'VBG',
                     'ADJP', 'JJR', 'NP', 'JJS', 'DT', 'FW', 'RBR', 'RBS',
                     'SBAR', 'RB']),
  'ADVP': ('left', ['RB', 'RBR', 'RBS', 'FW', 'ADVP', 'TO', 'CD', 'JJR', 'JJ',
                    'IN', 'NP', 'JJS', 'NN']),
  'CONJP': ('left', ['CC', 'RB', 'IN']),
  'FRAG': ('left', []),
  'INTJ': ('right', []),
  'LST': ('left', ['LS', ':']),
  'NAC': ('right', ['NN', 'NNS', 'NNP', 'NNPS', 'NP', 'NAC', 'EX', '$', 'CD',
                    'QP', 'PRP', 'VBG', 'JJ', 'JJS', 'JJR', 'ADJP', 'FW']),
  'PP': ('left', ['IN', 'TO', 'VBG', 'VBN', 'RP', 'FW']),
  'PRN': ('right', []),
  'PRT': ('left', ['RP']),
  'QP': ('right', ['$', 'IN', 'NNS', 'NN', 'JJ', 'RB', 'DT', 'CD', 'NCD', 'QP',
                   'JJR', 'JJS']),
  'RRC': ('left', ['VP', 'NP', 'ADVP', 'ADJP', 'PP']),
  'S': ('right', ['IN', 'VP', 'S', 'SBAR', 'ADJP', 'UCP', 'NP']),
  'SBAR': ('right', ['WHNP', 'WHPP', 'WHADVP', 'WHADJP', 'IN', 'DT', 'S', 'SQ',
                     'SINV', 'SBAR', 'FRAG']),
  'SBARQ': ('right', ['SQ', 'S', 'SINV', 'SBARQ', 'FRAG']),
  'SINV': ('right', ['VBZ', 'VBD', 'VBP', 'VB', 'MD', 'VP', 'S', 'SINV',
                     'ADJP', 'NP']),
  'SQ': ('right', ['VBZ', 'VBD', 'VBP', 'VB', 'MD', 'VP', 'SQ']),
  'UCP': ('left', []),
  'VP': ('right', ['VBD', 'VBN', 'MD', 'VBZ', 'VB', 'VBG', 'VBP', 'VP',
                   'ADJP', 'NN', 'NNS', 'NP']),
  'WHADJP': ('right', ['CC', 'WRB', 'JJ', 'ADJP']),
  'WHADVP': ('left', ['CC', 'WRB']),
  'WHNP': ('right', ['WDT', 'WP', 'WP$', 'WHADJP', 'WHPP', 'WHNP']),
  'WHPP': ('left', ['IN', 'TO', 'FW']),
  # Added by Kummerfeld:
  'NX': ('right', ['NN', 'NNS', 'NNP', 'NNPS', 'NP', 'NAC', 'EX', '$', 'CD',
                   'QP', 'PRP', 'VBG', 'JJ', 'JJS', 'JJR', 'ADJP', 'FW']),
  'X': ('right', ['NN', 'NNS', 'NNP', 'NNPS', 'NP', 'NAC', 'EX', '$', 'CD',
                  'QP', 'PRP', 'VBG', 'JJ', 'JJS', 'JJR', 'ADJP', 'FW']),
  'META': ('right', [])
}

def add_head(head_map, tree, head):
    if tree.__hash__:
        key = tree
    else:
        key = id(tree)
    head_map[key] = head

def get_head(head_map, tree):
    if tree.__hash__:
        key = tree
    else:
        key = id(tree)
    return head_map[key]

def first_search(tree, options, head_map):
    for subtree in tree:
        if (subtree.label() in options
            or get_head(head_map, subtree).label() in options):
            add_head(head_map, tree, get_head(head_map, subtree))
            return True
    return False

def last_search(tree, options, head_map):
    for i in xrange(len(tree) - 1, -1, -1):
        subtree = tree[i]
        if (subtree.label() in options
            or get_head(head_map, subtree).label() in options):
            add_head(head_map, tree, get_head(head_map, subtree))
            return True
    return False

def add_collins_NP(tree, head_map):
    for subtree in tree:
        collins_find_heads(subtree, head_map)
    # Original code had todo for: Extra special cases for NPs
    '''
    From Collins:

    Ignore the row for NPs -- I use a special set of rules for this. For these
    I initially remove ADJPs, QPs, and also NPs which dominate a possesive
    (tagged POS, e.g.  (NP (NP the man 's) telescope ) becomes
    (NP the man 's telescope)). These are recovered as a post-processing stage
    after parsing. The following rules are then used to recover the NP head:
    '''

    # Original code had todo for: handle NML properly

    last_child_head = get_head(head_map, tree[-1])
    if last_child_head.label() == 'POS':
        add_head(head_map, tree, last_child_head)
        return
    if last_search(tree, ['NN', 'NNP', 'NNPS', 'NNS', 'NX', 'POS', 'JJR'],
                   head_map):
        return
    if first_search(tree, ['NP', 'NML'], head_map):
        return
    if last_search(tree, ['$', 'ADJP', 'PRN'], head_map):
        return
    if last_search(tree, ['CD'], head_map):
        return
    if last_search(tree, ['JJ', 'JJS', 'RB', 'QP'], head_map):
        return
    add_head(head_map, tree, last_child_head)

def collins_find_heads(tree, head_map=None):
    '''
    Returns a table mapping all subtrees of the tree provided to their heads.
    ("Head" in this context means the node dominating only the head word.)
    If the tree provided is immutable (and therefore hashable), the keys will
    be the nodes themselves. Otherwise, the ids of the nodes will be used as
    keys. (Note that this means that the table will be invalidated by any
    changes to the tree.) It's probably advisable to simply convert the tree to
    an immutable tree before using this function.
    '''
    # TODO: Is it even a good idea to allow non-immutable trees?

    if head_map is None:
        head_map = {}

    # A word is its own head.
    if is_parent_of_leaf(tree):
        add_head(head_map, tree, tree)
        return head_map

    for subtree in tree:
        collins_find_heads(subtree, head_map)

    # If the label for this node is not in the table we either are at the
    #  bottom, are at an NP, or have an error.
    if tree.label() not in collins_mapping_table:
        if tree.label() in ['NP', 'NML']:
            add_collins_NP(tree, head_map)
        else:
            if tree.label() not in ['ROOT', 'TOP', 'S1', '']:
                logging.warn("Unknown label: %s" % tree.label())
            else:
                add_head(head_map, tree, get_head(head_map, tree[-1]))
        return head_map

    # Look through and take the first/last occurrence that matches
    info = collins_mapping_table[tree.label()]
    for label in info[1]:
        for i in range(len(tree)):
            if info[0] == 'right':
                i = len(tree) - i - 1
            subtree = tree[i]
            if (subtree.label() == label
                or get_head(head_map, subtree).label() == label):
                add_head(head_map, tree, get_head(head_map, subtree))
                return head_map

    # Final fallback
    if info[0] == 'left':
        add_head(head_map, tree, get_head(head_map, tree[0]))
    else: # right
        add_head(head_map, tree, get_head(head_map, tree[-1]))
    return head_map
