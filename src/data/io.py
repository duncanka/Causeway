from __future__ import absolute_import

from bidict._bidict import bidict
from gflags import FLAGS, DEFINE_bool, DEFINE_string, DuplicateFlagError
import io
import logging
import os
import numpy as np
import re

from iaa import stringify_connective
from util import recursively_list_files
from util.streams import read_stream_until, CharacterTrackingStreamWrapper
from data import (StanfordParsedSentence, Annotation, CausationInstance,
                  SentencesDocument, OverlappingRelationInstance)

try:
    DEFINE_bool('reader_binarize_degrees', True,
                'Whether to turn all degrees into "Facilitate" and "Inhibit"')
    DEFINE_string('reader_codec', 'utf-8',
                  'The encoding to assume for data files')
    DEFINE_bool('reader_gold_parses', False,
                'Whether to read .parse.gold files instead of .parse files for'
                ' sentence parses')
    DEFINE_bool('gold_parses_fallback', False,
                'If reader_gold_parses is True, falls back to automated parse'
                ' files instead of failing if gold parses are not found')
    DEFINE_bool('reader_directory_recurse', False,
                'Whether DirectoryReaders should recurse into their'
                ' subdirectories')
except DuplicateFlagError as e:
    logging.warn('Ignoring redefinition of flag %s' % e.flagname)


class DocumentStream(object):
    def __init__(self, filepath=None):
        self._file_stream = None
        if filepath:
            self.open(filepath)

    def close(self):
        if self._file_stream:
            self._file_stream.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, value, tb):
        self.close()


class DocumentWriter(DocumentStream):
    # TODO: change write and instance_complete to start_writing(doc),
    # incremental_write(instance), and finish_writing(doc). Will also eliminate
    # the write_all_instances thing.

    def open(self, filepath, mode='w'):
        self.close()
        self._file_stream = io.open(filepath, mode)

    def instance_complete(self, document, instance):
        pass

    def write(self, document):
        raise NotImplementedError


class InstancesDocumentWriter(DocumentWriter):
    '''
    Writer for documents where instances can be written one at a time. Such a
    writer must be used for sub-document incremental output.
    '''
    def write(self, document):
        pass # all the work happens in instance_complete

    def instance_complete(self, document, instance):
        self._write_instance(document, instance)
        self._file_stream.flush()

    def write_all_instances(self, document, instances_getter):
        all_instances = instances_getter(document)
        for instance in all_instances:
            self.instance_complete(document, instance)

    def _write_instance(self, document, instance):
        '''
        Writes a single instance to the current file. Must be overridden.
        '''
        raise NotImplementedError


class DocumentReader(DocumentStream):
    '''
    A document reader reads a file and produces a sequence of Documents. Often
    there is only one document per file, but there may be more. (For example, a
    reader may want to return every sentence as a separate "document.")
    '''

    def __iter__(self):
        next_instance = self.get_next()
        while next_instance is not None:
            yield next_instance
            next_instance = self.get_next()

    def open(self, filepath):
        self.close()
        self._file_stream = CharacterTrackingStreamWrapper(
            io.open(filepath, 'rb'), FLAGS.reader_codec)

    # TODO: convert into the more Pythonic paradigm of next()
    def get_next(self):
        raise NotImplementedError

    def get_all(self):
        return list(self)


class StanfordParsedSentenceReader(DocumentReader):
    '''
    Reads a single text document, along with pre-parsed Stanford parser output
    for that file. Returns one SentencesDocument of StanfordParsedSentences per
    file.
    '''
    def __init__(self, filepath=None):
        self._parse_file = None
        super(StanfordParsedSentenceReader, self).__init__(filepath)

    def open(self, filepath):
        super(StanfordParsedSentenceReader, self).open(filepath)
        base_path, _ = os.path.splitext(filepath)
        parse_file_name = base_path + '.parse'
        if FLAGS.reader_gold_parses:
            non_gold_file_name = parse_file_name
            parse_file_name += ".gold"
        try:
            self._parse_file = CharacterTrackingStreamWrapper(
                io.open(parse_file_name, 'rb'), FLAGS.reader_codec)
        except:
            if FLAGS.reader_gold_parses and FLAGS.gold_parses_fallback:
                logging.info("Falling back to non-gold parse for %s", filepath)
                self._parse_file = CharacterTrackingStreamWrapper(
                    io.open(non_gold_file_name, 'rb'), FLAGS.reader_codec)
            else:
                raise

    def close(self):
        super(StanfordParsedSentenceReader, self).close()
        if self._parse_file:
            self._parse_file.close()

    def get_next(self):
        sentences = []
        while True:
            next_sentence = self.get_next_sentence()
            if next_sentence is None: # end of file
                break
            sentences.append(next_sentence)

        if sentences: # There were some sentences in the file
            return SentencesDocument(self._file_stream.name, sentences)
        else:
            return None

    def get_next_sentence(self):
        # Read the next 3 blocks of the parse file.
        tokenized = self._parse_file.readline()
        if not tokenized: # empty string means we've hit the end of the file
            return None
        tokenized = tokenized.strip()
        tmp = self._parse_file.readline()
        assert not tmp.strip(), (
            'Invalid parse file: expected blank line after tokens: %s'
            % tokenized).encode('ascii', 'replace')

        lemmas = self._parse_file.readline()
        lemmas = lemmas.strip()
        assert lemmas, (
            'Invalid parse file: expected lemmas line after tokens: %s'
             % tokenized).encode('ascii', 'replace')
        tmp = self._parse_file.readline()
        assert not tmp.strip(), (
            'Invalid parse file: expected blank line after lemmas: %s'
            % lemmas).encode('ascii', 'replace')

        # If the sentence was unparsed, don't return a new StanfordParsedSentence for
        # it, but do advance the stream past the unparsed words.
        # NOTE: This relies on the printWordsForUnparsed flag we introduced to
        # the Stanford parser.
        if lemmas == '(())':
            self.__skip_tokens(tokenized, 'Ignoring unparsed sentence')
            return self.get_next()

        constituency_parse, double_newline_found = read_stream_until(
            self._parse_file, '\n\n')
        assert double_newline_found, (
            'Invalid parse file: expected blank line after constituency parse: %s'
            % constituency_parse).encode('ascii', 'replace')

        parse_lines = []
        tmp = self._parse_file.readline().strip()
        if not tmp:
            self.__skip_tokens(tokenized,
                               'Skipping sentence with empty dependency parse')
            return self.get_next()
        while tmp:
            parse_lines.append(tmp)
            tmp = self._parse_file.readline().strip()

        # Leaves file in the state where the final blank line after the edges
        # has been read. This also means that if there's a blank line at the end
        # of a file, it won't make us think there's another entry coming.

        # Now create the sentence from the read data + the text file.
        sentence = StanfordParsedSentence(
            tokenized, lemmas, constituency_parse, parse_lines,
            self._file_stream)
        assert (len(sentence.original_text) ==
                self._file_stream.character_position
                  - sentence.document_char_offset), (
            'Sentence length != offset difference: %s'
             % sentence.original_text).encode('ascii', 'replace')
        return sentence

    def __skip_tokens(self, tokenized, message):
        print '%s: %s' % (message, tokenized)
        for token in tokenized.split():
            unescaped = StanfordParsedSentence.unescape_token_text(token)
            _, found_token = read_stream_until(
                self._parse_file, unescaped, False, False)
            assert found_token, ('Skipped token not found: %s'
                                 % unescaped).encode('ascii', 'replace')


class DirectoryReader(DocumentReader):
    '''
    Reads all the Documents matching a set of regexes, using an underlying file
    reader. Thus, it will return many Documents for a single path.
    '''
    def __init__(self, file_regexes, base_reader):
        self._regexes = [re.compile(regex) for regex in file_regexes]
        self._base_reader = base_reader
        self._filenames = iter([])

    def open(self, dirpath):
        self.close()

        if not os.path.isdir(dirpath):
            raise IOError("No such directory: '%s" % dirpath)

        if FLAGS.reader_directory_recurse:
            self._filenames = recursively_list_files(dirpath)
        else:
            filenames = [os.path.join(dirpath, f) for f in os.listdir(dirpath)]
            self._filenames = (f for f in filenames if os.path.isfile(f))

        try:
            self.__open_next_file()
        except StopIteration:
            pass

    def close(self):
        self._base_reader.close()
        self._filenames = iter([])

    def get_next(self):
        # Start by seeing if our current file has any juice left.
        next_instance = self._base_reader.get_next()
        while not next_instance:
            try:
                self.__open_next_file()
                next_instance = self._base_reader.get_next()
            except StopIteration:
                self._filenames = iter([])
                return None
        return next_instance

    def __open_next_file(self):
        while True: # Eventually, we'll get a StopIteration if nothing matches.
            next_filename = self._filenames.next()
            for regex in self._regexes:
                if regex.match(os.path.basename(next_filename)):
                    self._base_reader.close()
                    self._base_reader.open(next_filename)
                    return


class CausalityStandoffReader(DocumentReader):
    '''
    Returns a Stanford-parsed SentencesDocument, with CausationInstances added
    to each sentence.
    '''
    FILE_PATTERN = r'.*\.ann$'

    def __init__(self, filepath=None):
        self.sentence_reader = StanfordParsedSentenceReader()
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
            instances_also_overlapping = []
            unused_arg_ids = set()
            self.__process_lines(lines, ids_to_annotations, ids_to_instances,
                                 instances_also_overlapping, unused_arg_ids,
                                 document)

            for to_duplicate, instance_type in instances_also_overlapping:
                to_duplicate.sentence.add_overlapping_instance(
                    instance_type, to_duplicate.connective, to_duplicate.arg0,
                    to_duplicate.arg1, to_duplicate.id, to_duplicate)

            for sentence in document:
                for ovl_instance in sentence.overlapping_rel_instances:
                    if ovl_instance.type is None:
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
                logging.warn("Count of lines to process has not shrunk after"
                             " recursion. Giving up on the following IDs: %s"
                             % ', '.join(ids_needed_to_reprocess))
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
            from_arg_id, to_arg_id = [arg_str.split(':')[1]
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
        annotation = Annotation(containing_sentence.document_char_offset,
                                annotation_offsets, text_str, line_id)
        ids_to_annotations[line_id] = annotation

        # Create the instance if necessary.
        annotation_type = type_and_indices_str[:first_space_idx]
        if annotation_type != 'Argument' and annotation_type != 'Note':
            is_noncausal = annotation_type == 'NonCausal'
            self.__raise_warning_if(
                annotation_type not in CausationInstance.CausationTypes
                and not is_noncausal,
                "Skipping text annotation with invalid causation type")
            try:
                connective = containing_sentence.find_tokens_for_annotation(
                    annotation)
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
        else:
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
                    instance.type = overlapping_type
                else:
                    instances_also_overlapping.append((instance,
                                                       overlapping_type))
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
                        instance.sentence.find_tokens_for_annotation(
                            annotation))
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
