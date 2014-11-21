import logging
import os
import re

from util import streams, recursively_list_files
from data import ParsedSentence, Annotation, CausationInstance

# TODO: convert readers into the more Pythonic paradigm of essentially acting
# like generators.

class Reader(object):
    def __init__(self):
        self._file_stream = None

    def open(self, filepath):
        self.close()
        self._file_stream = open(filepath, 'r')

    def close(self):
        if self._file_stream:
            self._file_stream.close()

    def get_next(self):
        raise NotImplementedError

    def get_all(self):
        instances = []
        instance = self.get_next()
        while instance is not None:
            instances.append(instance)
            instance = self.get_next()
        return instances


class SentenceReader(Reader):
    def __init__(self):
        super(SentenceReader, self).__init__()
        self._parse_file = None

    def open(self, filepath):
        super(SentenceReader, self).open(filepath)
        base_path, _ = os.path.splitext(filepath)
        self._parse_file = open(base_path + '.parse', 'r')

    def close(self):
        super(SentenceReader, self).close()
        if self._parse_file:
            self._parse_file.close()

    def get_next(self):
        if not self._parse_file:
            return None

        # Read the next 3 blocks of the parse file.
        tokenized = self._parse_file.readline()
        if not tokenized: # empty string means we've hit the end of the file
            return None
        tokenized = tokenized.strip()
        tmp = self._parse_file.readline()
        assert not tmp.strip(), (
            'Invalid parse file: expected blank line after tokens: %s'
            % tokenized)

        lemmas = self._parse_file.readline()
        lemmas = lemmas.strip()
        assert lemmas, (
            'Invalid parse file: expected lemmas line after tokens: %s'
             % tokenized)
        tmp = self._parse_file.readline()
        assert not tmp.strip(), (
            'Invalid parse file: expected blank line after lemmas: %s' % lemmas)

        # If the sentence was unparsed, don't return a new ParsedSentence for
        # it, but do advance the stream past the unparsed words.
        # NOTE: This relies on the printWordsForUnparsed flag we introduced to
        # the Stanford parser.
        if lemmas == '(())':
            self.__skip_tokens(tokenized, 'Ignoring unparsed sentence')
            return self.get_next()

        parse_lines = []
        tmp = self._parse_file.readline().strip()
        if not tmp:
            self.__skip_tokens(tokenized, 'Skipping sentence with empty parse')
            return self.get_next()
        while tmp:
            parse_lines.append(tmp)
            tmp = self._parse_file.readline().strip()

        # Leaves file in the state where the final blank line after the edges
        # has been read. This also means that if there's a blank line at the end
        # of a file, it won't make us think there's another entry coming.

        # Now create the sentence from the read data + the text file.
        sentence = ParsedSentence(
            tokenized, lemmas, parse_lines, self._file_stream)
        assert (len(sentence.original_text) == self._file_stream.tell()
                - sentence.document_char_offset), \
            'Sentence length != offset difference: %s' % sentence.original_text
        return sentence

    def __skip_tokens(self, tokenized, message):
        print '%s: %s' % (message, tokenized)
        for token in tokenized.split():
            unescaped = ParsedSentence.unescape_token_text(token)
            _, found_token = streams.read_stream_until(
                self._parse_file, unescaped, False, False)
            assert found_token, 'Skipped token not found: %s' % unescaped

class DirectoryReader(Reader):
    def __init__(self, file_regexes, base_reader):
        self._regexes = [re.compile(regex) for regex in file_regexes]
        self._base_reader = base_reader
        self._filenames = iter([])

    def open(self, dirpath):
        if not os.path.isdir(dirpath):
            raise IOError("No such directory: '%s" % dirpath)

        self._filenames = recursively_list_files(dirpath)
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
            except StopIteration:
                self._filenames = iter([])
                return None
            next_instance = self._base_reader.get_next()
        return next_instance

    def __open_next_file(self):
        while True: # Eventually, we'll get a StopIteration if nothing matches.
            next_filename = self._filenames.next()
            for regex in self._regexes:
                if regex.match(os.path.basename(next_filename)):
                    self._base_reader.close()
                    self._base_reader.open(next_filename)
                    return


class StandoffReader(Reader):
    ''' Returns ParsedSentence instances, with CausationInstances added. '''
    def __init__(self):
        super(StandoffReader, self).__init__()
        self.sentence_reader = SentenceReader()
        self.instances = []
        self.iterator = iter([])

    def open(self, filepath):
        super(StandoffReader, self).open(filepath)
        base_path, _ = os.path.splitext(filepath)
        self.sentence_reader.open(base_path + '.txt')
        self.__read_all_instances()

    def close(self):
        super(StandoffReader, self).close()
        # self.sentence_reader gets closed immediately after opening, so we
        # don't need to bother closing it again.
        self.instances = []
        self.iterator = iter([])

    def get_next(self):
        return next(self.iterator, None)

    def __read_all_instances(self):
        self.instances = self.sentence_reader.get_all()
        self.sentence_reader.close()

        lines = self._file_stream.readlines()
        if not lines:
            logging.warn("No annotations found")
            # Don't close the reader: we still want to return the sentences,
            # even if they have no causality annotations.
        else:
            ids_to_annotations = {}
            ids_to_instances = {}
            self.__process_lines(lines, ids_to_annotations, ids_to_instances)

        self.iterator = iter(self.instances)

    @staticmethod
    def __raise_warning_if(condition, message):
        if condition:
            raise UserWarning(message)

    def __process_lines(self, lines, ids_to_annotations, ids_to_instances):
        lines_to_reprocess = []
        ids_to_reprocess = set()
        ids_needed_to_reprocess = set()

        for line in lines:
            try:
                stripped = line.strip()
                line_parts = stripped.split('\t')
                if len(line_parts) < 2:
                    raise UserWarning(
                        "Ignoring line not formatted as ID, tab, content")

                line_id = line_parts[0]
                if line_id[0] == 'T': # it's an annotation span
                    self.__process_text_annotation(
                        line, line_parts, ids_to_annotations, ids_to_instances,
                        lines_to_reprocess, ids_to_reprocess,
                        ids_needed_to_reprocess)
                elif line_id[0] == 'A': # it's an attribute of an event (degree)
                    self.__process_attribute(
                        line, line_parts, ids_to_annotations, ids_to_instances,
                        lines_to_reprocess, ids_to_reprocess,
                        ids_needed_to_reprocess)
                elif line_id[0] == 'E': # it's an event
                    self.__process_event(
                        line, line_parts, ids_to_annotations,
                        ids_to_instances, lines_to_reprocess, ids_to_reprocess,
                        ids_needed_to_reprocess)
                elif line_id[0] == '#': # skip annotator notes lines silently
                    continue
                else:
                    logging.info("Ignoring annotation line: %s" % stripped)

            except UserWarning as e:
                logging.warn('%s (Line: %s)' % (e.message, stripped))
                return

        # There is no possibility of cyclical relationships in our annotation
        # scheme, so it's OK to just assume that with each pass we'll reduce the
        # set of IDs that need to be added.
        if lines_to_reprocess:
            recurse = True
            for id_needed in ids_needed_to_reprocess:
                # Any ID that was referenced before being defined must be
                # defined somewhere -- either we've seen a definition since
                # then, or it's something we're intending to define on the next
                # pass.
                if (not ids_to_annotations.has_key(id_needed) and
                    not ids_to_instances.has_key(id_needed) and
                    id_needed not in ids_to_reprocess):
                    logging.warn(
                        "ID %s is referenced, but is not defined anywhere. "
                        "Ignoring all lines that depend on it." % id_needed)
                    recurse = False
            if recurse:
                self.__process_lines(lines_to_reprocess, ids_to_annotations,
                                     ids_to_instances)

    def __process_text_annotation(self, line, line_parts, ids_to_annotations,
                                  ids_to_instances, lines_to_reprocess,
                                  ids_to_reprocess, ids_needed_to_reprocess):
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
        containing_sentence = StandoffReader.find_containing_sentence(
            annotation_offsets, self.instances, line)
        self.__raise_warning_if(
            containing_sentence is None,
            "Skipping annotation for which no sentence could be found")
        annotation = Annotation(containing_sentence.document_char_offset,
                                annotation_offsets, text_str, line_id)
        ids_to_annotations[line_id] = annotation

        # Create the instance if necessary.
        annotation_type = type_and_indices_str[:first_space_idx]
        if annotation_type != 'Argument' and annotation_type != 'Note':
            self.__raise_warning_if(
                annotation_type not in CausationInstance.CausationTypes,
                "Skipping text annotation with invalid causation type")
            instance = CausationInstance(containing_sentence)
            ids_to_instances[line_id] = instance
            instance.connective = (
                containing_sentence.find_tokens_for_annotation(annotation))
            containing_sentence.add_causation_instance(instance)

    def __process_attribute(self, line, line_parts, ids_to_annotations,
                            ids_to_instances, lines_to_reprocess,
                            ids_to_reprocess, ids_needed_to_reprocess):
        self.__raise_warning_if(
            len(line_parts) != 2,
            "Skipping attribute line lacking 2 tab-separated entries")
        line_id = line_parts[0]
        attr_parts = line_parts[1].split(' ')
        self.__raise_warning_if(
            len(attr_parts) != 3,
            "Skipping attribute line lacking 3 space-separated components")
        self.__raise_warning_if(
            attr_parts[0] != "Degree",
            "Skipping attribute line with unrecognized attribute")

        _, id_to_modify, degree = attr_parts
        try:
            degree_index = CausationInstance.Degrees.index(degree)
            ids_to_instances[id_to_modify].degree = degree_index
        except ValueError:
            raise UserWarning('Skipping attribute line with invalid degree')
        except KeyError:
            lines_to_reprocess.append(line)
            ids_to_reprocess.add(line_id)
            ids_needed_to_reprocess.add(id_to_modify)

    def __process_event(self, line, line_parts, ids_to_annotations,
                        ids_to_instances, lines_to_reprocess,
                        ids_to_reprocess, ids_needed_to_reprocess):
        self.__raise_warning_if(len(line_parts) != 2,
            "Skipping event line that does not have 2 tab-separated entries")
        line_id = line_parts[0]
        args = line_parts[1].split(' ')
        self.__raise_warning_if(
            len(args) < 1,
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
            sentence = instance.source_sentence
            # There can be a numerical suffix on the end of the name of the
            # edge. Since we're generally assuming well-formed data, we don't
            # check that there's only one of each.
            for arg_type, arg_id in split_args[1:]:
                annotation = ids_to_annotations[arg_id]
                annotation_tokens = sentence.find_tokens_for_annotation(
                    annotation)
                if arg_type.startswith('Cause'):
                    instance.cause = annotation_tokens
                elif arg_type.startswith('Effect'):
                    instance.effect = annotation_tokens
                else:
                    raise UserWarning('Skipping event with invalid arg types')

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

        if result is None:
            raise Exception

        return result
