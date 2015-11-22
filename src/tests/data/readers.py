from __future__ import absolute_import

from mock import patch, DEFAULT, call
import unittest

from data.readers import Reader


class ReaderTest(unittest.TestCase):
    def test_context_manager(self):
        with patch.multiple(Reader, open=DEFAULT, get_next=DEFAULT,
                            close=DEFAULT):
            with Reader('/test/path.txt') as reader:
                reader.get_next()

            reader.open.assert_called_once_with('/test/path.txt')
            reader.close.assert_called_with()

    def test_iterator(self):
        true_lines = ['Hi', 'there', None]
        with patch.multiple(Reader, open=DEFAULT, get_next=DEFAULT,
                            close=DEFAULT) as MockReader:
            MockReader['get_next'].side_effect = true_lines
            with Reader('/test/path.txt') as reader:
                lines = [line for line in reader]

            self.assertEqual(true_lines[:-1], lines)
            self.assertEqual([call(), call(), call()],
                             reader.get_next.call_args_list)

    def test_get_all(self):
        true_lines = ['Hi', 'there', None]
        with patch.multiple(Reader, open=DEFAULT, get_next=DEFAULT,
                            close=DEFAULT) as MockReader:
            MockReader['get_next'].side_effect = true_lines
            reader = Reader('/test/path.txt')
            lines = reader.get_all()

            self.assertEqual(true_lines[:-1], lines)
            self.assertEqual([call(), call(), call()],
                             reader.get_next.call_args_list)
