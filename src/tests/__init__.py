import gflags
import inspect
import numpy as np
from os import path
import sys
import unittest

gflags.FLAGS([]) # Prevent UnparsedFlagAccessError


def get_resources_dir():
    # UGLY HACK to figure out where the caller is located, find the closest
    # enclosing "tests" directory, and grab its "resources" directory.
    stack_i = 0
    frame = inspect.stack()[stack_i][0]
    while frame.f_globals['__file__'] == __file__:
        stack_i += 1
        frame = inspect.stack()[stack_i][0]

    caller_path = frame.f_globals['__file__']
    caller_root_tests_module_path = caller_path
    head, tail = path.split(caller_root_tests_module_path)
    # Keep splitting off the final directory until the final directory is called
    # "tests". Then, use the directory that "tests" was split from.
    while tail != 'tests':
        if not tail:
            raise IOError('No "tests" directory above ' + caller_path)
        caller_root_tests_module_path = head
        head, tail = path.split(caller_root_tests_module_path)
    return path.join(caller_root_tests_module_path, 'resources')

def get_documents_from_file(reader_type, subdir, filename, resources_dir=None):
    if not resources_dir:
        resources_dir = get_resources_dir()
    reader = reader_type()
    reader.open(path.join(resources_dir, subdir, filename))
    documents = reader.get_all()
    reader.close()
    return documents

def get_sentences_from_file(reader_type, subdir, filename, resources_dir=None):
    sentences = []
    for document in get_documents_from_file(reader_type, subdir, filename,
                                            resources_dir):
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
