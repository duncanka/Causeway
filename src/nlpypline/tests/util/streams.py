from io import StringIO, BytesIO, SEEK_END
import unittest

from nlpypline.util.streams import (CharacterTrackingStreamWrapper, eat_whitespace,
                          is_at_eof, read_stream_until)


class CharacterTrackingStreamWrapperTest(unittest.TestCase):
    data_str = ("Guinea has been in turmoil for more than a month,\nas labor"
        " unions and other civic groups have demanded that the ailing leader,"
        " Lansana Cont\xc3\xa9, step aside.")

    @classmethod
    def setUpClass(cls):
        cls.data_decoded = cls.data_str.decode('utf-8')
        cls.data_decoded_split = cls.data_decoded.split('\n')
        cls.data_decoded_split[0] += '\n'

    def setUp(self):
        self.document = CharacterTrackingStreamWrapper(BytesIO(self.data_str),
                                                       'utf-8')

    def test_basic_read(self):
        read_data = self.document.read()
        self.assertEqual(read_data, self.data_decoded)
        self.assertEqual(len(read_data), self.document.character_position)

    def test_readline(self):
        read_data = self.document.readline()
        self.assertEqual(read_data, self.data_decoded_split[0])
        self.assertEqual(len(read_data), self.document.character_position)

    def test_readlines(self):
        self.assertEqual(self.document.readlines(), self.data_decoded_split)
        self.assertEqual(len(self.data_decoded),
                         self.document.character_position)

    def test_iteration(self):
        self.assertEqual([line for line in self.document],
                         self.data_decoded_split)
        self.assertEqual(len(self.data_decoded),
                         self.document.character_position)

    def test_seek_to_ends(self):
        self.document.seek(0, SEEK_END)
        self.assertEqual(len(self.data_decoded),
                         self.document.character_position)

        # Test seeking to start
        self.document.seek(0)
        self.assertEqual(0, self.document.character_position)

        # Test seeking to end from the middle
        self.document.read(41)
        self.document.seek(0, SEEK_END)
        self.assertEqual(len(self.data_decoded),
                         self.document.character_position)

    def test_seek_to_saved(self):
        self.document.read(41)
        self.assertEqual(41, self.document.character_position)

        # Try seeking repeatedly in case seeking back & forth messes things up.
        for _ in range(2):
            saved = self.document.tell()
            self.assertEqual(self.document.readline(), u'a month,\n')
            self.assertEqual(50, self.document.character_position)
            self.document.seek(saved)
            self.assertEqual(41, self.document.character_position)
            self.assertEqual(saved, self.document.tell())

        # Now do the same for a range that includes a Unicode character.
        self.document.readline() # move back to end of line
        # Read to start of 'aside', which starts at character 147 in Unicode
        read_data = self.document.read(97)
        self.assertEqual(self.data_decoded_split[1][:97], read_data)
        self.assertEqual(147, self.document.character_position)
        for _ in range(2):
            saved = self.document.tell()
            self.assertEqual(self.document.readline(), u'aside.')
            self.assertEqual(len(self.data_decoded),
                             self.document.character_position)
            self.document.seek(saved)
            self.assertEqual(147, self.document.character_position)
            self.assertEqual(saved, self.document.tell())


class StreamFunctionsTest(unittest.TestCase):
    data_str = u'This  is a test\nof \n stream manipulation.'

    def setUp(self):
        self.stream = StringIO(self.data_str)

    def test_eat_whitespace(self):
        for token, expected_ws in zip(self.data_str.split(),
                                      ['  ', ' ', ' ', '\n', ' \n ', ' ', '']):
            self.assertEqual(self.stream.read(len(token)), token)
            ws = eat_whitespace(self.stream, True)
            self.assertEqual(expected_ws, ws,
                             ('Did not find expected ws "%s" after "%s"'
                              % (expected_ws, token)))

    def test_is_at_eof(self):
        self.assertFalse(is_at_eof(self.stream))

        self.stream.read(10)
        self.assertFalse(is_at_eof(self.stream))

        saved_position = self.stream.tell()
        self.stream.read() # read to end
        end_position = self.stream.tell()
        self.assertTrue(is_at_eof(self.stream))

        self.stream.seek(saved_position)
        self.assertFalse(is_at_eof(self.stream))

        self.stream.seek(end_position)
        self.assertTrue(is_at_eof(self.stream))

    def test_read_stream_until_success(self):
        FIRST_WORD = self.data_str[:5]
        data, result = read_stream_until(self.stream, FIRST_WORD)
        self.assertEqual(data, FIRST_WORD)
        self.assertTrue(result)
        self.assertEqual(self.stream.tell(), len(data))

        remaining_1st_line = self.data_str.split('\n')[0]
        remaining_1st_line = remaining_1st_line[len(data):] + '\n'
        data, result = read_stream_until(self.stream, '\n')
        self.assertEqual(data, remaining_1st_line)
        self.assertTrue(result)
        self.assertEqual(self.stream.tell(),
                         len(remaining_1st_line) + len(FIRST_WORD))

    def test_read_stream_until_failure(self):
        data, result = read_stream_until(self.stream, 'q')
        self.assertEqual(data, self.data_str)
        self.assertFalse(result)
        self.assertEqual(self.stream.tell(), len(self.data_str))

if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
