import numpy as np
from os import path
import unittest

def get_resources_dir():
    return path.join(path.dirname(__file__), 'resources')

def get_documents_from_file(reader_type, subdir, filename):
        reader = reader_type()
        reader.open(path.join(get_resources_dir(), subdir, filename))
        documents = reader.get_all()
        reader.close()
        return documents

def get_sentences_from_file(reader_type, subdir, filename):
    sentences = []
    for document in get_documents_from_file(reader_type, subdir, filename):
        sentences.extend(document.sentences)
    return sentences


class NumpyAwareTestCase(unittest.TestCase):
    def assertArraysEqual(self, array1, array2):
        self.assertEqual(
            type(array1), np.ndarray,
            'array1 is not an array (actual type: %s)' % type(array1))
        self.assertEqual(
            type(array2), np.ndarray,
            'array2 is not an array (actual type: %s)' % type(array2))

        self.assertEqual(array1.shape, array2.shape,
                         'Array shapes do not match (%s vs %s)'
                         % (array1.shape, array2.shape))

        if not array1.dtype.isbuiltin or not array2.dtype.isbuiltin:
            self.assertEqual(
                array1.dtype, array2.dtype,
                'Incompatible dtypes: %s vs %s' % (array1.dtype, array2.dtype))

        comparison = (array1 == array2)
        if comparison.all():
            return
        else:
            num_differing = comparison.size - np.count_nonzero(comparison)
            msg = ('Arrays differ at %d locations\n%s\n\nvs.\n\n%s'
                   % (num_differing, array1, array2))
            self.fail(msg)
