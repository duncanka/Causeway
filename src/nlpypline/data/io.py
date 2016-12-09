from __future__ import absolute_import

from gflags import FLAGS, DEFINE_bool, DEFINE_string, DuplicateFlagError
import io
import logging
import os
import re

from nlpypline.data import StanfordParsedSentence, SentencesDocument
from nlpypline.util import recursively_list_files
from nlpypline.util.streams import (read_stream_until,
                                    CharacterTrackingStreamWrapper)


try:
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
    logging.warn('Ignoring flag redefinitions; assuming module reload')


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
        next_document = self.get_next()
        while next_document is not None:
            yield next_document
            next_document = self.get_next()

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
    def __init__(self, filepath=None, sentence_class=StanfordParsedSentence):
        if not issubclass(sentence_class, StanfordParsedSentence):
            raise TypeError("StanfordParsedSentenceReader can only parse to"
                            " subclasses of StanfordParsedSentence")
        self._parse_file = None
        self.sentence_class = sentence_class
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

        # If the sentence was unparsed, don't return a new sentence object for
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
        sentence = self.sentence_class(
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
