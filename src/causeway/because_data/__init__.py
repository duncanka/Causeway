"""
Data structures and readers/writers for dealing with data from the BECauSE
corpus.
"""

from __future__ import absolute_import, print_function

from bidict import bidict
from collections import defaultdict, deque
from copy import copy, deepcopy
from gflags import FLAGS, DuplicateFlagError, DEFINE_bool
import logging
from nltk.tree import ImmutableParentedTree
from nltk.util import flatten
import numpy as np
import os
from scipy.sparse.lil import lil_matrix

from nlpypline.data import Annotation, Token, StanfordParsedSentence
from nlpypline.data.io import (DocumentReader, StanfordParsedSentenceReader,
                               InstancesDocumentWriter)
from nlpypline.util import listify, Enum, make_getter, make_setter, Object
from textwrap import TextWrapper


try:
    DEFINE_bool('reader_binarize_degrees', True,
                'Whether to turn all degrees into "Facilitate" and "Inhibit"')
    DEFINE_bool('reader_ignore_overlapping', False,
                'Whether, when reading causality data, instances with an'
                ' accompanying overlapping relation should be ignored')
except DuplicateFlagError as e:
    logging.warn('Ignoring flag redefinitions; assuming module reload')


class CausewaySentence(StanfordParsedSentence):
    def __init__(self, *args, **kwargs):
        super(CausewaySentence, self).__init__(*args, **kwargs)
        self.causation_instances = []
        self.overlapping_rel_instances = []

    def add_causation_instance(self, *args, **kwargs):
        instance = CausationInstance(self, *args, **kwargs)
        self.causation_instances.append(instance)
        return instance

    def add_overlapping_instance(self, *args, **kwargs):
        instance = OverlappingRelationInstance(self, *args, **kwargs)
        self.overlapping_rel_instances.append(instance)
        return instance

    def dep_to_ptb_tree_string(self):
        # Collapsed dependencies can have cycles, so we need to avoid infinite
        # recursion.
        visited = set()
        def convert_node(node, incoming_arc_label):
            # If we've already visited the node before, don't recurse on it --
            # just re-output its own string. In the vast majority of cases,
            # this will match the patterns just fine, since they're only
            # matching heads anyway. And since we're duplicating the node name,
            # we'll know later what real node matched.
            recurse = node not in visited
            visited.add(node)
            lemma = self.escape_token_text(node.lemma)
            node_str = '(%s_%d %s %s' % (lemma, node.index,
                                         incoming_arc_label, node.pos)

            for child_arc_label, child in sorted(
                self.get_children(node),
                key=lambda pair: pair[1].start_offset):
                if recurse and child_arc_label != 'ref':
                    node_str += ' ' + convert_node(child, child_arc_label)
            node_str += ')'
            return node_str

        return '(ROOT %s)' % convert_node(
            self.get_children(self.tokens[0], 'root')[0], 'root')

    def substitute_dep_ptb_graph(self, ptb_str):
        '''
        Returns a copy of the sentence object, whose edge graph has been
        replaced by the one represented in `ptb_str`. Uses
        `shallow_copy_with_sentences_fixed` to get a mostly shallow copy, but
        with correctly parented CausationInstance and Token objects.
        '''
        edge_graph, edge_labels, excluded_edges = (
            self._sentence_graph_from_ptb_str(ptb_str, len(self.tokens)))
        new_sentence = self.shallow_copy_with_sentences_fixed(self)
        new_sentence.edge_graph = edge_graph
        new_sentence.edge_labels = edge_labels
        new_sentence._initialize_graph(excluded_edges)
        return new_sentence

    @staticmethod
    def _sentence_graph_from_ptb_str(ptb_str, num_tokens):
        # We need to have num_tokens provided here, or else we won't know for
        # sure how big the graph should be. (There can be tokens missing from
        # the graph, and even if there aren't it would take more processing
        # than it's worth to find the max node index in the PTB tree.)
        tree = ImmutableParentedTree.fromstring(ptb_str)
        edge_graph = lil_matrix((num_tokens, num_tokens), dtype='float')
        edge_labels = {}
        excluded_edges = []

        def convert_node(parent_index, node):
            # Node index is whatever's after the last underscore.
            node_label = node.label()
            node_index = int(node_label[node_label.rindex('_') + 1:])
            edge_label = node[0]  # 0th child is always edge label
            if edge_label in StanfordParsedSentence.DEPTH_EXCLUDED_EDGE_LABELS:
                excluded_edges.append((parent_index, node_index))
            else:
                edge_graph[parent_index, node_index] = 1.0
            edge_labels[parent_index, node_index] = edge_label

            for child in node[2:]: # Skip edge label (child 0) & POS (child 1).
                convert_node(node_index, child)

        for root_child in tree:
            convert_node(0, root_child) # initial parent index is 0 for root
        return edge_graph.tocsr(), edge_labels, excluded_edges

    def get_auxiliaries_string(self, head):
        # If it's not a copular construction and it's a noun phrase, the whole
        # argument is a noun phrase, so the notion of tense doesn't apply.
        copulas = self.get_children(head, 'cop')
        if not copulas and head.pos in Token.NOUN_TAGS:
            return '<NOM>'

        auxiliaries = self.get_children(head, 'aux')
        passive_auxes = self.get_children(head, 'auxpass')
        auxes_plus_head = auxiliaries + passive_auxes + copulas + [head]
        auxes_plus_head.sort(key=lambda token: token.start_offset)

        CONTRACTION_DICT = {
            "'s": 'is',
             "'m": 'am',
             "'d": 'would',
             "'re": 'are',
             'wo': 'will', # from "won't"
             'ca': 'can' # from "can't"
        }
        aux_token_strings = []
        for token in auxes_plus_head:
            if token is head:
                aux_token_strings.append(token.pos)
            else:
                if token in copulas:
                    aux_token_strings.append('COP.' + token.pos)
                else:
                    aux_token_strings.append(
                         CONTRACTION_DICT.get(token.lowered_text,
                                              token.lowered_text))

        return '_'.join(aux_token_strings)

    @staticmethod
    def shallow_copy_with_sentences_fixed(sentence):
        '''
        Creates a shallow copy of sentence, but with causation_instances and
        Tokens on the new object set to shallow copies of the original objects,
        so that they can know their correct source sentence.
        '''
        cls = sentence.__class__
        new_sentence = cls.__new__(cls)
        new_sentence.__dict__.update(sentence.__dict__)

        new_sentence.tokens = []
        for token in sentence.tokens:
            new_token = object.__new__(token.__class__)
            new_token.__dict__.update(token.__dict__)
            new_token.parent_sentence = new_sentence
            new_sentence.tokens.append(new_token)

        new_sentence.causation_instances = []
        for causation_instance in sentence.causation_instances:
            new_instance = copy(causation_instance)
            new_instance.sentence = new_sentence
            # Update connective/cause/effect tokens to the ones that point to
            # the new sentence object.
            if causation_instance.connective:
                new_instance.connective = [new_sentence.tokens[t.index] for t
                                           in causation_instance.connective]
            if causation_instance.cause:
                new_instance.cause = [new_sentence.tokens[t.index] for t
                                      in causation_instance.cause]
            if causation_instance.effect:
                new_instance.effect = [new_sentence.tokens[t.index] for t
                                       in causation_instance.effect]
            new_sentence.causation_instances.append(new_instance)
        return new_sentence


class _RelationInstance(object):
    _num_args = 2

    def __init__(self, source_sentence, connective, arg0=None, arg1=None,
                 rel_type=None, annotation_id=None):
        assert source_sentence is not None
        for token in listify(connective) + listify(arg0) + listify(arg1):
            if token is None:
                continue
            # Parent sentence is no longer always the source sentence
            # assert token.parent_sentence is source_sentence

        self.sentence = source_sentence
        self.connective = connective
        self.arg0 = arg0
        self.arg1 = arg1
        self.type = rel_type
        self.id = annotation_id

    @classmethod
    def get_arg_types(klass, convert=False):
        arg_types = ['arg%d' % i for i in range(klass._num_args)]
        if convert:
            return [klass.arg_names[name] for name in arg_types]
        else:
            return arg_types

    def get_args(self):
        return [getattr(self, arg_name) for arg_name in self.get_arg_types()]

    def get_named_args(self, convert=False):
        return {arg_name: getattr(self, arg_name)
                for arg_name in self.get_arg_types(convert)}

    def get_argument_heads(self, head_sort_key=None):
        """
        head_sort_key is a function that takes an argument and returns a key by
        which to sort it. If this parameter is provided, argument heads are
        returned in the resulting order.
        """
        arg_heads = [self.sentence.get_head(arg) if arg else None
                     for arg in self.get_args()]
        if head_sort_key:
            arg_heads.sort(key=head_sort_key)
        return arg_heads

    __wrapper = TextWrapper(80, subsequent_indent='    ', break_long_words=True)

    @staticmethod
    def pprint(instance):
        # TODO: replace with same code as IAA?
        connective = ' '.join([t.original_text for t in instance.connective])
        named_args = instance.get_named_args(convert=True)
        arg_strings = [
             u'{arg_name}={txt}'.format(
                arg_name=arg_name,
                txt=u' '.join([t.original_text for t in annotation]
                              if annotation else [u'<None>']))
             for arg_name, annotation in sorted(named_args.iteritems())]
        self_str = u'{typename}(connective={conn}, {args}, type={type})'.format(
            typename=instance.__class__.__name__, conn=connective,
            args=u', '.join(arg_strings), type=instance._get_type_str())
        return u'\n'.join(_RelationInstance.__wrapper.wrap(self_str))

    def get_interpretable_type(self):
        if self.type is not None:
            return self._types[self.type]
        else:
            return "UNKNOWN"

    def _get_type_str(self):
        return self.get_interpretable_type()

    def __repr__(self):
        return self.pprint(self).encode('utf-8')

    arg_names = bidict({'arg0': 'arg0', 'arg1': 'arg1'})


class CausationInstance(_RelationInstance):
    Degrees = Enum(['Facilitate', 'Enable', 'Disentail', 'Inhibit'])
    CausationTypes = Enum(['Consequence', 'Motivation','Purpose', 'Inference'])
    _types = CausationTypes
    _num_args = 3

    def __init__(self, source_sentence, degree=None, causation_type=None,
                 connective=None, cause=None, effect=None, means=None,
                 annotation_id=None):
        if degree is None:
            degree = len(self.Degrees)
        if causation_type is None:
            degree = len(self.CausationTypes)

        super(CausationInstance, self).__init__(source_sentence, connective,
                                                cause, effect, causation_type,
                                                annotation_id)
        self.degree = degree
        self.arg2 = means

    # Map argument attribute names to arg_i attributes.
    arg_names = bidict({'arg0': 'cause', 'arg1': 'effect', 'arg2': 'means'})

for arg_attr_name in ['cause', 'effect', 'means']:
    underlying_property_name = CausationInstance.arg_names.inv[arg_attr_name]
    getter = make_getter(underlying_property_name)
    setter = make_setter(underlying_property_name)
    setattr(CausationInstance, arg_attr_name, property(getter, setter))


class OverlappingRelationInstance(_RelationInstance):
    RelationTypes = Enum(['Temporal', 'Correlation', 'Hypothetical',
                          'Obligation_permission', 'Creation_termination',
                          'Extremity_sufficiency', 'Context'])
    _types = RelationTypes

    def __init__(self, source_sentence, rel_type=None, connective=None,
                 arg0=None, arg1=None, annotation_id=None,
                 attached_causation=None):
        if rel_type is None:
            rel_type = set() # overlapping rel can have multiple types

        all_args = locals().copy()
        del all_args['self']
        del all_args['attached_causation']
        super(OverlappingRelationInstance, self).__init__(**all_args)

        self.attached_causation = attached_causation

    def get_interpretable_type(self):
        if self.type:
            return set(self._types[t] for t in self.type)
        else:
            return set(['UNKNOWN'])

    def _get_type_str(self):
        return '+'.join(self.get_interpretable_type())


# Useful shorthand. It's not really a type, but it can be used like one to
# create new reader objects.
CausewaySentenceReader = lambda: StanfordParsedSentenceReader(
    sentence_class=CausewaySentence)


class CausalityStandoffReader(DocumentReader):
    '''
    Returns a Stanford-parsed SentencesDocument, with CausationInstances added
    to each sentence.
    '''
    FILE_PATTERN = r'.*\.ann$'

    def __init__(self, filepath=None):
        self.sentence_reader = CausewaySentenceReader()
        super(CausalityStandoffReader, self).__init__(filepath)

    def open(self, filepath):
        super(CausalityStandoffReader, self).open(filepath)
        base_path, _ = os.path.splitext(filepath)
        self.sentence_reader.open(base_path + '.txt')

    def close(self):
        super(CausalityStandoffReader, self).close()
        # self.sentence_reader gets closed immediately after opening, so we
        # don't need to bother closing it again.
        self.sentence_reader.close()

    def get_next(self):
        document = self.sentence_reader.get_next()
        if not document:
            return None

        lines = self._file_stream.readlines()
        if not lines:
            logging.warn("No annotations found in file %s"
                         % self._file_stream.name)
            # Don't close the reader: we still want to return the sentences,
            # even if they have no causality annotations.
        else:
            ids_to_annotations = {}
            ids_to_instances = {}
            # Map of causal instances to their overlapping relations
            instances_also_overlapping = defaultdict(set)
            unused_arg_ids = set()
            self.__process_lines(lines, ids_to_annotations, ids_to_instances,
                                 instances_also_overlapping, unused_arg_ids,
                                 document)

            for to_duplicate, types in instances_also_overlapping.items():
                to_duplicate.sentence.add_overlapping_instance(
                    types, to_duplicate.connective, to_duplicate.arg0,
                    to_duplicate.arg1, to_duplicate.id, to_duplicate)

            for sentence in document:
                for ovl_instance in sentence.overlapping_rel_instances:
                    if ovl_instance.type is None:
                        from causeway.because_data.iaa import (
                            stringify_connective)
                        logging.warn(
                            "No relation type for non-causal instance %s (%s)",
                            ovl_instance.id, stringify_connective(ovl_instance))

        return document

    @staticmethod
    def __raise_warning_if(condition, message):
        if condition:
            raise UserWarning(message)

    def __process_lines(self, lines, ids_to_annotations, ids_to_instances,
                        instances_also_overlapping, unused_arg_ids, document,
                        prev_line_count=np.inf):
        lines_to_reprocess = []
        ids_to_reprocess = set()
        ids_needed_to_reprocess = set()

        for line in lines:
            try:
                stripped = line.strip()
                line_parts = stripped.split('\t')
                self.__raise_warning_if(
                    len(line_parts) < 2,
                    "Ignoring line not formatted as ID, tab, content")

                line_id = line_parts[0]
                if line_id[0] == 'T': # it's an annotation span
                    self.__process_text_annotation(
                        line, line_parts, ids_to_annotations, ids_to_instances,
                        lines_to_reprocess, ids_to_reprocess,
                        ids_needed_to_reprocess, unused_arg_ids, document)
                elif line_id[0] == 'A': # it's an event attribute
                    self.__process_attribute(
                        line, line_parts, ids_to_annotations, ids_to_instances,
                        instances_also_overlapping, lines_to_reprocess,
                        ids_to_reprocess, ids_needed_to_reprocess)
                elif line_id[0] == 'E': # it's an event
                    self.__process_event(
                        line, line_parts, ids_to_annotations,
                        ids_to_instances, lines_to_reprocess, ids_to_reprocess,
                        ids_needed_to_reprocess, unused_arg_ids)
                elif line_parts[1].startswith('Coref'):
                    self.__process_coref_line(
                        line, line_parts, ids_to_annotations, unused_arg_ids,
                        lines_to_reprocess, ids_to_reprocess,
                        ids_needed_to_reprocess)
                # skip annotator notes silently
                elif line_id[0] == '#':
                    continue
                else:
                    raise UserWarning("Ignoring unrecognized annotation line")

            except UserWarning as e:
                logging.warn('%s (File: %s; Line: %s)'
                             % (e.message, self._file_stream.name, stripped))

        # There is no possibility of cyclical relationships in our annotation
        # scheme, so it's OK to just assume that with each pass we'll reduce
        # the set of IDs that need to be added.
        recurse = False
        if lines_to_reprocess:
            if len(lines_to_reprocess) >= prev_line_count:
                logging.warn("Count of lines to process in %s has not shrunk"
                             " after recursion. Giving up on the following IDs:"
                             " %s" % (self._file_stream.name,
                                      ', '.join(ids_needed_to_reprocess)))
                return
            for id_needed in ids_needed_to_reprocess:
                # Any ID that was referenced before being defined must be
                # defined somewhere -- either we've seen a definition since
                # then, or it's something we're intending to define on the next
                # pass.
                if (ids_to_annotations.has_key(id_needed) or
                    ids_to_instances.has_key(id_needed) or
                    id_needed in ids_to_reprocess):
                    recurse = True
                else:
                    logging.warn(
                        "ID %s is referenced, but is not defined anywhere. "
                        "Ignoring all lines that depend on it. (File: %s)"
                        % (id_needed, self._file_stream.name))
        if recurse:
            self.__process_lines(lines_to_reprocess, ids_to_annotations,
                                 ids_to_instances, instances_also_overlapping,
                                 unused_arg_ids, document, len(lines))
        else:
            for arg_id in unused_arg_ids:
                logging.warn('Unused argument: %s: "%s" (file: %s)'
                             % (arg_id, ids_to_annotations[arg_id].text,
                                self._file_stream.name))

    def __process_coref_line(self, line, line_parts, ids_to_annotations,
                             unused_arg_ids, lines_to_reprocess,
                             ids_to_reprocess, ids_needed_to_reprocess):
        try:
            _line_id, coref_str = line_parts
            coref_args = coref_str.split()[1:]
            _from_arg_id, to_arg_id = [arg_str.split(':')[1]
                                       for arg_str in coref_args]
        except ValueError:
            logging.warn('Skipping incorrectly formatted coref line.'
                         ' (Line: %s)' % line.rstrip())
            return

        try:
            unused_arg_ids.remove(to_arg_id)
        except KeyError:
            # Being unable to mark an arg ID as used means either that it's
            # not in the document at all; that it's in the document but we
            # haven't read it yet; or that it's in the document and it's already
            # been marked used.
            #
            # For the first or second cases, we want to flag this line for
            # reanalysis, and if we end up not finding it (1st case) we'll
            # eventually complain. For the third case, we don't need to do
            # anything further.
            if to_arg_id not in ids_to_annotations:
                lines_to_reprocess.append(line)
                ids_needed_to_reprocess.add(to_arg_id)

    def __process_text_annotation(self, line, line_parts, ids_to_annotations,
                                  ids_to_instances, lines_to_reprocess,
                                  ids_to_reprocess, ids_needed_to_reprocess,
                                  unused_arg_ids, document):
        try:
            line_id, type_and_indices_str, text_str = line_parts
        except ValueError:
            logging.warn(("Skipping annotation span line that doesn't have 3 "
                          "tab-separated entries. (Line: %s)") % line)
            return

        self.__raise_warning_if(
            ' ' not in type_and_indices_str,
            'Skipping annotation span line with no space in type/index string')
        first_space_idx = type_and_indices_str.index(' ')
        indices_str = type_and_indices_str[first_space_idx + 1:]
        annotation_offsets = []
        for index_pair_str in indices_str.split(';'):
            index_pair = [int(index) for index in index_pair_str.split(' ')]
            self.__raise_warning_if(
                len(index_pair) != 2,
                'Skipping annotation span line without 2 indices')
            annotation_offsets.append(tuple(index_pair))

        # Create the new annotation.
        containing_sentence = CausalityStandoffReader.find_containing_sentence(
            annotation_offsets, document.sentences, line)
        self.__raise_warning_if(
            containing_sentence is None,
            "Skipping annotation for which no sentence could be found")
        annotation = Annotation(annotation_offsets, text_str, line_id)
        ids_to_annotations[line_id] = annotation

        # Create the instance if necessary.
        annotation_type = type_and_indices_str[:first_space_idx]
        if annotation_type != 'Argument' and annotation_type != 'Note':
            is_noncausal = annotation_type == 'NonCausal'
            self.__raise_warning_if(
                annotation_type not in CausationInstance.CausationTypes
                and not is_noncausal,
                "Skipping text annotation with invalid causation type")
            if is_noncausal and FLAGS.reader_ignore_overlapping:
                return

            try:
                connective = self._find_tokens_for_annotation(
                    containing_sentence, annotation)
                if is_noncausal:
                    instance = containing_sentence.add_overlapping_instance(
                        connective=connective)
                else:
                    instance = containing_sentence.add_causation_instance(
                        connective=connective)
                ids_to_instances[line_id] = instance
            except ValueError as e: # No tokens found for annotation
                raise UserWarning(e.message)

        elif annotation_type == 'Argument':
            unused_arg_ids.add(line_id)

    def _find_tokens_for_annotation(self, sentence, annotation):
        tokens = []
        cross_sentence = Object() # wrap in object to allow access by add_token
        cross_sentence.status = False
        def add_token(token):
            tokens.append(token)
            if token.parent_sentence is not sentence:
                cross_sentence.status = True

        tokens_to_search = []
        double_prev_sentence = (sentence.previous_sentence.previous_sentence
                                if sentence.previous_sentence else None)
        double_next_sentence = (sentence.next_sentence.next_sentence
                                if sentence.next_sentence else None)
        for tokens_sentence in (double_prev_sentence,
                                sentence.previous_sentence, sentence,
                                sentence.next_sentence, double_next_sentence):
            if tokens_sentence:
                tokens_to_search.extend(tokens_sentence.tokens[1:]) # skip ROOT
        tokens_iter = iter(tokens_to_search)
        next_token = tokens_iter.next()
        try:
            for start, end in annotation.offsets:
                prev_token = None
                while next_token.get_start_offset_doc() < start:
                    prev_token = next_token
                    next_token = tokens_iter.next()
                if next_token.get_start_offset_doc() != start:
                    warning = ("Start of annotation %s in file %s does not"
                               " correspond to a token start"
                               % (annotation.id, sentence.source_file_path))
                    if prev_token and prev_token.get_end_offset_doc() >= start:
                        add_token(prev_token)
                        warning += '; the token it bisects has been appended'
                    logging.warn(warning)
                # We might have grabbed a whole additional token just because
                # of an annotation that included a final space, so make sure
                # next_token really is in the annotation span before adding it.
                if next_token.get_start_offset_doc() < end:
                    add_token(next_token)

                while next_token.get_end_offset_doc() < end:
                    prev_token = next_token
                    next_token = tokens_iter.next()
                    if next_token.get_start_offset_doc() < end:
                        add_token(next_token)
                if next_token.get_end_offset_doc() != end:
                    warning = ("End of annotation %s in file %s does not"
                               " correspond to a token start"
                               % (annotation.id, sentence.source_file_path))
                    # If we appended the next token, that means the index
                    # brought us into the middle of the next word.
                    if tokens[-1] is next_token:
                        warning += '; the token it bisects has been appended'
                    logging.warn(warning)

            if cross_sentence.status:
                logging.warn("%s: Annotation %s crosses sentence boundary: %s",
                             sentence.source_file_path, annotation.id,
                             annotation.text)

            # TODO: Should we check to make sure the annotation text is right?
            return tokens

        except StopIteration:
            raise ValueError("Annotation %s couldn't be matched against tokens!"
                             " Ignoring..." % annotation.offsets)

    def __process_attribute(self, line, line_parts, ids_to_annotations,
                            ids_to_instances, instances_also_overlapping,
                            lines_to_reprocess, ids_to_reprocess,
                            ids_needed_to_reprocess):

        self.__raise_warning_if(
            len(line_parts) != 2,
            "Skipping attribute line lacking 2 tab-separated entries")
        line_id = line_parts[0]
        attr_parts = line_parts[1].split()

        attr_type = attr_parts[0]
        if attr_type == 'Degree':
            self.__raise_warning_if(
                len(attr_parts) != 3,
                "Skipping attribute line lacking 3 space-separated components")

            _, id_to_modify, degree = attr_parts
            try:
                if FLAGS.reader_binarize_degrees:
                    if degree == 'Enable':
                        degree = 'Facilitate'
                    elif degree == 'Disentail':
                        degree = 'Inhibit'
                degree_index = getattr(CausationInstance.Degrees, degree)
                ids_to_instances[id_to_modify].degree = degree_index
            except ValueError:
                raise UserWarning('Skipping attribute line with invalid degree')
            except KeyError:
                lines_to_reprocess.append(line)
                ids_to_reprocess.add(line_id)
                ids_needed_to_reprocess.add(id_to_modify)
        else: # It's an overlapping relation attribute.
            if FLAGS.reader_ignore_overlapping:
                logging.info("Ignoring attribute: %s", line)
                return

            self.__raise_warning_if(
                len(attr_parts) != 2,
                "Skipping attribute line lacking 2 space-separated components")

            id_to_modify = attr_parts[1]
            attr_type = attr_type.replace('-', '_')
            try:
                overlapping_type = getattr(
                    OverlappingRelationInstance.RelationTypes, attr_type)
                instance = ids_to_instances[id_to_modify]
                if isinstance(instance, OverlappingRelationInstance):
                    instance.type.add(overlapping_type)
                else:
                    instances_also_overlapping[instance].add(overlapping_type)
            except AttributeError:
                raise UserWarning(
                    "Skipping attribute line with unrecognized attribute: %s"
                    % attr_type)
            except KeyError:
                lines_to_reprocess.append(line)
                ids_to_reprocess.add(line_id)
                ids_needed_to_reprocess.add(id_to_modify)

    def __process_event(self, line, line_parts, ids_to_annotations,
                        ids_to_instances, lines_to_reprocess,
                        ids_to_reprocess, ids_needed_to_reprocess,
                        unused_arg_ids):
        self.__raise_warning_if(len(line_parts) != 2,
            "Skipping event line that does not have 2 tab-separated entries")
        line_id = line_parts[0]
        args = line_parts[1].split()
        # TODO: update this to handle zero-arg instances?
        self.__raise_warning_if(
            len(args) < 2,
            'Skipping event line that does not have at least 1 arg')
        split_args = [arg.split(':') for arg in args]
        self.__raise_warning_if(
            not all([len(arg) == 2 for arg in split_args]),
            "Skipping event line whose argument doesn't have 2 components")

        # We know we at least have 1 arg, and that each arg has 2 components,
        # because we verified both of those above.
        causation_type, connective_id = split_args[0]
        try:
            causation_type_index = CausationInstance.CausationTypes.index(
                causation_type)
        except ValueError:
            if causation_type == 'NonCausal':
                causation_type_index = -1
            else:
                raise UserWarning('Skipping invalid causation type: %s'
                                  % causation_type)

        id_needed = None
        try:
            instance = ids_to_instances[connective_id]
            for arg_id in [arg[1] for arg in split_args]:
                if not ids_to_annotations.has_key(arg_id):
                    id_needed = arg_id
                    break
        except KeyError:
            id_needed = line_id
            # Don't even bother processing the rest of the line if we're just
            # going to have to reprocess it later.

        if id_needed:
            lines_to_reprocess.append(line)
            ids_to_reprocess.add(line_id)
            ids_needed_to_reprocess.add(id_needed)
        else:
            # There can be a numerical suffix on the end of the name of the
            # edge. Since we're generally assuming well-formed data, we don't
            # check that there's only one of each.
            for arg_type, arg_id in split_args[1:]:
                annotation = ids_to_annotations[arg_id]
                try:
                    annotation_tokens = (
                        self._find_tokens_for_annotation(
                            instance.sentence, annotation))
                except ValueError as e:
                    raise UserWarning(e.message)

                try:
                    try:
                        setattr(instance, arg_type.lower(), annotation_tokens)
                    except AttributeError:
                        # This could be an annotation whose arc label started
                        # out as a duplicate and therefore got an extra numeral
                        # on the end. Just in case, retry without the last
                        # character of the arg type.
                        setattr(instance, arg_type[:-1].lower(),
                                annotation_tokens)
                except AttributeError:
                    raise UserWarning('Skipping event with invalid arg type %s'
                                      % arg_type)

                try:
                    unused_arg_ids.remove(arg_id)
                except KeyError:
                    # Don't worry about this -- just means the argument was
                    # used twice, so it already got removed.
                    pass

            if causation_type_index != -1: # it's not a NonCausal instance
                instance.type = causation_type_index
            instance.id = line_id
            # Add the event ID as an alias of the instance.
            ids_to_instances[line_id] = instance

    @staticmethod
    def find_containing_sentence(offsets, sentences, line):
        result = None
        last_sentence = None
        first_start = offsets[0][0]
        for sentence in sentences:
            if sentence.document_char_offset > first_start:
                result = last_sentence
                break
            last_sentence = sentence

        # It could still be in the last sentence.
        if result is None and last_sentence is not None:
            if (last_sentence.document_char_offset +
                len(last_sentence.original_text)) > first_start:
                result = last_sentence

        return result


class CausalityStandoffWriter(InstancesDocumentWriter):
    def __init__(self, filepath=None, initial_char_offset=0):
        super(CausalityStandoffWriter, self).__init__(filepath)
        self._reset()
        self.initial_char_offset = initial_char_offset

    def write(self, document):
        # The real work was already done in instance_complete.
        # Now reset internals.
        self.reset()

    def _reset(self):
        self._next_event_id = 1
        self._next_annotation_id = 1
        self._next_attribute_id = 1
        self._objects_to_ids = bidict()

    @staticmethod
    def _get_annotation_bounds(tokens):
        sentence = tokens[0].parent_sentence
        token_iterator = iter(sorted(tokens, key=lambda t: t.index))
        bounds = []
        next_token = next(token_iterator)

        try:
            span_start = next_token.start_offset
            span_end = next_token.end_offset
            prev_token = next_token
            next_token = next(token_iterator)

            while True:
                while next_token.index == prev_token.index + 1:
                    # If there's a line break in between the previous token and
                    # the upcoming one, create an artificial fragment like brat.
                    if '\n' in sentence.original_text[prev_token.end_offset:
                                                      next_token.start_offset]:
                        break
                    span_end = next_token.end_offset # extend span
                    prev_token = next_token
                    next_token = next(token_iterator)

                # Now we've reached the end of a contiguous span. Append bounds
                # and start off another span.
                bounds.append((span_start, span_end))

                span_start = next_token.start_offset
                span_end = next_token.end_offset
                prev_token = next_token
                next_token = next(token_iterator)

        except StopIteration:
            bounds.append((span_start, span_end))

        return bounds

    def _get_bounds_and_text_strings(self, tokens, annotation_type_str):
        sentence = tokens[0].parent_sentence
        bounds = CausalityStandoffWriter._get_annotation_bounds(tokens)
        bounds_str = ';'.join(
            ['%d %d' % tuple(sentence.document_char_offset + index
                             - self.initial_char_offset for index in bound_pair)
             for bound_pair in bounds])
        bounds_str = ' '.join([annotation_type_str, bounds_str])
        text_str = ' '.join(
            [sentence.original_text[span_start:span_end]
             for span_start, span_end in bounds])
        return (bounds_str, text_str)

    def _make_id_for(self, obj, next_id_attr_name, id_prefix):
        if id(obj) in self._objects_to_ids:
            raise KeyError('Attempted to write object %s twice' % obj)

        try:
            if obj.id is not None:
                self._objects_to_ids[id(obj)] = obj.id
                return obj.id
        except AttributeError: # No id attribute
            pass

        # No saved ID; make up a new one, making sure not to clash with any that
        # have already been assigned.
        next_id_num = getattr(self, next_id_attr_name)
        new_id = '%s%d' % (id_prefix, next_id_num)
        while new_id in self._objects_to_ids.inv:
            next_id_num += 1
            new_id = '%s%d' % (id_prefix, next_id_num)
        # We're now using this ID. Next valid one is this one + 1.
        setattr(self, next_id_attr_name, next_id_num + 1)

        self._objects_to_ids[id(obj)] = new_id
        try:
            obj.id = new_id
        except AttributeError: # this wasn't an instance object with an ID
            pass
        return new_id

    def _make_attribute_id(self):
        # Attributes can never be shared, so don't worry about reuse with
        # self._objects_to_ids.
        attr_id = 'A%d' % self._next_attribute_id
        self._next_attribute_id += 1
        return attr_id

    def _write_line(self, *line_components):
        self._file_stream.write(u'\t'.join(line_components))
        self._file_stream.write(u'\n')

    def _write_argument(self, arg_tokens):
        if not arg_tokens:
            return

        try:
            arg_id = self._make_id_for(arg_tokens, '_next_annotation_id', 'T')
            bounds_str, text_str = self._get_bounds_and_text_strings(arg_tokens,
                                                                     'Argument')
        except KeyError: # Already written. Not a problem; args are often shared
            return

        self._write_line(arg_id, bounds_str, text_str)

    def _get_arg_string(self, arg_name, arg):
        if arg is None:
            return ''
        arg_id = self._objects_to_ids[id(arg)]
        return ':'.join([arg_name, arg_id])

    def _write_event(self, instance, instance_type_name):
        event_id = self._make_id_for(instance, '_next_event_id', 'E')
        connective_id = self._make_id_for(instance.connective,
                                          '_next_annotation_id', 'T')

        bounds_str, text_str = self._get_bounds_and_text_strings(
            instance.connective, instance_type_name)
        self._write_line(connective_id, bounds_str, text_str)

        arg_strings = [
            self._get_arg_string(instance.arg_names[arg_type].title(),
                                 getattr(instance, arg_type))
            for arg_type in instance.get_arg_types()]
        event_component_strings = (
            [':'.join([instance_type_name, connective_id])]
            + [arg for arg in arg_strings if arg]) # skip blank args
        self._write_line(event_id, ' '.join(event_component_strings))
        return event_id

    def _write_causation(self, instance):
        for arg in instance.get_args():
            self._write_argument(arg)

        instance_type = CausationInstance.CausationTypes[instance.type]
        event_id = self._write_event(instance, instance_type)

        # Write degree if it's set.
        if instance.degree is not None:
            degree_attr_id = self._make_attribute_id()
            degree_string = ' '.join(['Degree', event_id,
                                    CausationInstance.Degrees[instance.degree]])
            self._write_line(degree_attr_id, degree_string)

    def _write_overlapping(self, instance):
        if instance.type is None:
            logging.warn("Skipping instance with no type: %s", instance)
            return

        if instance.attached_causation is not None:
            event_id = self._objects_to_ids[id(instance.attached_causation)]
        else:
            self._write_argument(instance.arg0)
            self._write_argument(instance.arg1)
            event_id = self._write_event(instance, 'NonCausal')

        ovl_attr_id = self._make_attribute_id()
        relation_type = OverlappingRelationInstance.RelationTypes[instance.type]
        relation_type = relation_type.replace('_', '-')
        ovl_attr_string = ' '.join([relation_type, event_id])
        self._write_line(ovl_attr_id, ovl_attr_string)

    def _write_instance(self, document, sentence):
        for causation_instance in sentence.causation_instances:
            self._write_causation(causation_instance)

        for overlapping_instance in sentence.overlapping_rel_instances:
            self._write_overlapping(overlapping_instance)


class CausalityOracleTransitionWriter(InstancesDocumentWriter):
    def _write_instance(self, document, sentence):
        tokens = [token for token in sentence.tokens[1:]] # skip ROOT

        # Print sentence-initial line with tokens and POS tags.
        print(u', '.join(u'/'.join([t.original_text.replace(' ', ''), t.pos])
                         for t in tokens),
              file=self._file_stream)

        # Initialize state. lambda_1 is unexamined tokens to the left of the
        # current token; lambda_2 is examined tokens to the left; and likewise
        # for lambda_4 and lambda_3, respectively, to the right.
        self.lambda_1 = []
        self.lambda_2 = deque() # we'll be appending to the left end
        self.lambda_3 = []
        # We'll be moving stuff off of the left end of lambda_4.
        self.lambda_4 = deque(tokens)
        self.lambdas = [self.lambda_1, self.lambda_2, self.lambda_3,
                        self.lambda_4]
        self.rels = []
        self._last_op = None

        connectives_to_instances = defaultdict(list)
        for causation in sentence.causation_instances:
            first_conn_token = causation.connective[0]
            connectives_to_instances[first_conn_token].append(causation)

        # Make sure the instances for each token are sorted by order of
        # appearance.
        for _, causations in connectives_to_instances.iteritems():
            causations.sort(key=lambda instance: tuple(t.index for t in
                                                       instance.connective))

        for current_token in tokens:
            instance_under_construction = None
            token_instances = connectives_to_instances[current_token]
            if token_instances: # some connective starts with this token
                instance_under_construction = self._compare_with_conn(
                    current_token, True, token_instances,
                    instance_under_construction)
                self._compare_with_conn(current_token, False, token_instances,
                                        instance_under_construction)
                self._write_transition(current_token, 'SHIFT')
            else:
                self._write_transition(current_token, 'NO-CONN')

            if current_token is not tokens[-1]:
                self.lambda_1.extend(self.lambda_2)
                self.lambda_2.clear()
                self.lambda_1.append(current_token)
                if self.lambda_3: # we processed some right-side tokens
                    # Skip copy of current token.
                    self.lambda_4.extend(self.lambda_3[1:])
                    # If we didn't use del here, we'd have to reconstruct
                    # self.lambdas.
                    del self.lambda_3[:]
                else: # current_token was a no-conn
                    self.lambda_4.popleft()
        self._file_stream.write(u'\n') # Final blank line

        (self.lambda_1, self.lambda_2, self.lambda_3, self.lambda_4,
         self.lambdas, self.rels) = [None] * 6 # Reset; release memory

    def _do_split(self, current_token, last_modified_arg, token_to_compare,
                  instance_under_construction):
        self._write_transition(current_token, 'SPLIT')

        # Figure out where in the connective the token we're replacing is.
        conn_token_index = None
        for i, conn_token in enumerate(instance_under_construction.connective):
            if conn_token.lemma == token_to_compare.lemma:
                conn_token_index = i
        if conn_token_index is None:
            logging.warn("Didn't find a shared word when splitting connective;"
                         " sharing only the first word")
            conn_token_index = 1
        arg_cutoff_index = instance_under_construction.connective[
            conn_token_index].index

        instance_under_construction = deepcopy(instance_under_construction)
        self.rels.append(instance_under_construction)

        # We need to know which tokens to keep from the argument we were
        # building when we encountered this new connective token. We assume that
        # we should keep any token preceding the connective fragment we're
        # replacing.
        new_argument = [t for t in getattr(instance_under_construction,
                                           last_modified_arg)
                        if t.index < arg_cutoff_index]
        setattr(instance_under_construction, last_modified_arg,
                new_argument)

        instance_under_construction.connective = (
            instance_under_construction.connective[:conn_token_index])
        instance_under_construction.connective.append(token_to_compare)
        # other_connective_tokens.remove(token_to_compare)
        return instance_under_construction

    def _compare_with_conn(self, current_token, dir_is_left,
                           connective_instances, instance_under_construction):
        if dir_is_left:
            arc_direction = 'LEFT'
            first_uncompared_index = -1
            compared = self.lambda_2
            uncompared = self.lambda_1
        else:
            arc_direction = 'RIGHT'
            first_uncompared_index = 0
            compared = self.lambda_3
            uncompared = self.lambda_4

        conn_instance_index = 0
        conn_instance = connective_instances[conn_instance_index]
        other_connective_tokens = set(flatten(
            [i.connective for i in connective_instances[1:]]))
        other_connective_tokens -= set(conn_instance.connective)
        last_modified_arc_type = None
        while uncompared:
            token_to_compare = uncompared[first_uncompared_index]

            # First, see if we should split. But don't split on leftward tokens.
            if (not dir_is_left and token_to_compare in other_connective_tokens
                and self._last_op != 'SPLIT'):
                instance_under_construction = self._do_split(
                    current_token, last_modified_arc_type, token_to_compare,
                    instance_under_construction)
                # Move to next
                conn_instance_index += 1
                conn_instance = connective_instances[conn_instance_index]
                # Leave current token to be compared with new connective.
            else:
                # If there's a fragment, record it first, before looking at the
                # args. (The fragment word might still be part of an arg.)
                if (token_to_compare is not current_token
                    and self._last_op not in  # no fragments after splits/frags
                        ['SPLIT',  "CONN-FRAG-{}".format(arc_direction)]
                    and token_to_compare in conn_instance.connective):
                    self._write_transition(current_token,
                                           "CONN-FRAG-{}".format(arc_direction))
                    instance_under_construction.connective.append(
                        token_to_compare)

                arcs_to_add = []
                new_instance = False
                for arc_type in ['cause', 'effect', 'means']:
                    argument = getattr(conn_instance, arc_type, None)
                    if argument is not None and token_to_compare in argument:
                        arcs_to_add.append(arc_type)
                        if instance_under_construction is None:
                            instance_under_construction = CausationInstance(
                                conn_instance.sentence, cause=[], effect=[],
                                means=[], connective=[current_token])
                            new_instance = True
                        # TODO: This will do odd things if there's ever a SPLIT
                        # interacting with a multiple-argument arc.
                        last_modified_arc_type = arc_type
                if arcs_to_add:
                    trans = "{}-ARC({})".format(
                        arc_direction, ','.join(arc_type.title()
                                                for arc_type in arcs_to_add))
                    self._write_transition(current_token, trans)
                    if new_instance:
                        self.rels.append(instance_under_construction)
                    for arc_type in arcs_to_add:
                        getattr(instance_under_construction, arc_type).append(
                            token_to_compare)
                else:
                    self._write_transition(current_token,
                                           "NO-ARC-{}".format(arc_direction))
                    if instance_under_construction is None:
                        instance_under_construction = CausationInstance(
                            conn_instance.sentence, cause=[], effect=[],
                            means=[], connective=[current_token])
                        self.rels.append(instance_under_construction)

                if dir_is_left:
                    compared.appendleft(uncompared.pop())
                else:
                    compared.append(uncompared.popleft())

        return instance_under_construction # make update visible

    def _write_transition(self, current_token, transition):
        stringified_lambdas = [self._stringify_token_list(l)
                               for l in self.lambdas]
        state_line = u"{} {} {token} {} {}".format(
            *stringified_lambdas, token=self._stringify_token(current_token))
        rels_line = self._stringify_rels()
        for line in [state_line, rels_line, unicode(transition)]:
            print(line, file=self._file_stream)
        self._last_op = transition

    def _stringify_token(self, token):
        return u'{}-{}'.format(token.original_text.replace(' ', ''),
                               token.index)

    def _stringify_token_list(self, token_list):
        token_strings = [self._stringify_token(t) for t in token_list]
        return u'[{}]'.format(u', '.join(token_strings))

    def _stringify_rels(self):
        instance_strings = [
            u'{}({}, {}, {})'.format(u'/'.join([self._stringify_token(c)
                                                for c in instance.connective]),
                                 self._stringify_token_list(instance.cause),
                                 self._stringify_token_list(instance.effect),
                                 self._stringify_token_list(instance.means))
            for instance in self.rels]
        return u'{{{}}}'.format(u', '.join(instance_strings))
