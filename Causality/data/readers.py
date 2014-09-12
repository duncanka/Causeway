import os
import re
from util import streams, recursively_list_files
from data import ParsedSentence

class Reader(object):
    def __init__(self):
        self.file_stream = None

    def open(self, filepath):
        self.file_stream = open(filepath, 'r')

    def close(self):
        self.file_stream.close()

    def get_next(self):
        raise NotImplementedError

    def get_all(self):
        instances = []
        instance = self.get_next()
        while instance is not None:
            instances.append(instance)
        return instances


class SentenceReader(Reader):
    def __init__(self):
        self.last_opened_path = None
        self.parse_file = None

    def open(self, filepath):
        super(SentenceReader, self).open(filepath)
        (base_path, _) = os.path.splitext(filepath)
        self.parse_file = open(base_path + '.parse', 'r')

    def close(self):
        super(SentenceReader, self).close()
        self.parse_file.close()
        
    def get_next(self):
        if not self.parse_file:
            return None

        # Read the next 3 blocks of the parse file.
        tokenized = self.parse_file.readline()
        if not tokenized: # empty string means we've hit the end of the file
            return None
        tokenized = tokenized.strip()
        tmp = self.parse_file.readline()
        assert not tmp.strip(), (
            'Invalid parse file: expected blank line after tokens: %s' 
            % tokenized)

        lemmas = self.parse_file.readline()
        lemmas = lemmas.strip()
        assert lemmas, (
            'Invalid parse file: expected lemmas line after tokens: %s'
             % tokenized)
        tmp = self.parse_file.readline()
        assert not tmp.strip(), (
            'Invalid parse file: expected blank line after lemmas: %s' % lemmas)
        
        # If the sentence was unparsed, don't return a new ParsedSentence for it,
        # but do advance the stream past the unparsed words.
        # NOTE: This relies on the printWordsForUnparsed flag we introduced to
        # the Stanford parser.
        if lemmas == '(())':
            self.__skip_tokens(tokenized, 'Ignoring unparsed sentence')
            return self.get_next()

        parse_lines = []
        tmp = self.parse_file.readline().strip()
        if not tmp:
            self.__skip_tokens(tokenized, 'Skipping sentence with empty parse')
            return self.get_next()
        while tmp:
            parse_lines.append(tmp)
            tmp = self.parse_file.readline().strip()

        # Leaves file in the state where the final blank line after the edges has
        # been read. This also means that if there's a blank line at the end of a
        # file, it won't make us think there's another entry coming.

        # Now create the sentence from the read data + the text file.
        sentence = ParsedSentence(tokenized, lemmas, parse_lines, self.file_stream)
        assert (len(sentence.original_text) == self.file_stream.tell()
                - sentence.document_char_offset), \
            'Sentence length != offset difference: %s' % sentence.original_text
        return sentence

    def __skip_tokens(self, tokenized, message):
        print '%s: %s' % (message, tokenized)
        for token in tokenized.split():
            unescaped = ParsedSentence.unescape_token_text(token)
            _, found_token = streams.read_stream_until(
                self.parse_file, unescaped, False, False)
            assert found_token, 'Skipped token not found: %s' % unescaped
