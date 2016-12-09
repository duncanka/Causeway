from __future__ import absolute_import

from mock import patch, DEFAULT, call
import unittest

from nlpypline.data.io import DocumentReader, StanfordParsedSentenceReader
from tests import get_documents_from_file


class DocumentReaderTest(unittest.TestCase):
    def test_context_manager(self):
        with patch.multiple(DocumentReader, open=DEFAULT, get_next=DEFAULT,
                            close=DEFAULT):
            with DocumentReader('/test/path.txt') as reader:
                reader.get_next()

            reader.open.assert_called_once_with('/test/path.txt')
            reader.close.assert_called_with()

    def test_iterator(self):
        # Normally the reader would return a list of Documents, not strings, but
        # this is good enough for testing.
        true_strings = ['Hi', 'there', None]
        with patch.multiple(DocumentReader, open=DEFAULT, get_next=DEFAULT,
                            close=DEFAULT) as MockReader:
            MockReader['get_next'].side_effect = true_strings
            with DocumentReader('/test/path.txt') as reader:
                lines = [line for line in reader]

            self.assertEqual(true_strings[:-1], lines)
            self.assertEqual([call(), call(), call()],
                             reader.get_next.call_args_list)

    def test_get_all(self):
        true_strings = ['Hi', 'there', None]
        with patch.multiple(DocumentReader, open=DEFAULT, get_next=DEFAULT,
                            close=DEFAULT) as MockReader:
            MockReader['get_next'].side_effect = true_strings
            reader = DocumentReader('/test/path.txt')
            lines = reader.get_all()

            self.assertEqual(true_strings[:-1], lines)
            self.assertEqual([call(), call(), call()],
                             reader.get_next.call_args_list)


class StanfordParsedSentenceReaderTest(unittest.TestCase):
    # TODO: add more rigorous testing
    def test_reads_document(self):
        documents = get_documents_from_file(
            StanfordParsedSentenceReader, 'DataTest', 'data_test.txt')
        self.assertEqual(1, len(documents))
        self.assertEqual(5, len(documents[0].sentences))


# TODO: add Oracle transition writer tests