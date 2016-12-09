from gflags import DEFINE_string, FLAGS, DuplicateFlagError
import logging
from os import path
import numpy as np

from util import NameDictionary

try:
    DEFINE_string('senna_dir', '/home/jesse/Documents/Work/Research/senna/',
                  'Directory containing the SENNA installation')
except DuplicateFlagError as e:
    logging.warn('Ignoring flag redefinitions; assuming module reload')


class SennaEmbeddings(object):
    def __init__(self, senna_dir=None):
        self.words_dictionary = NameDictionary()
        
        if not senna_dir:
            senna_dir = FLAGS.senna_dir
        logging.info('Reading SENNA embeddings from %s...' % senna_dir)
        words_path = path.join(senna_dir, 'hash', 'words.lst')
        vectors_path = path.join(senna_dir, 'embeddings', 'embeddings.txt')

        with open(words_path, 'r') as words_file:
            for word_line in words_file:
                self.words_dictionary.insert(word_line.rstrip())

        self.embeddings = np.loadtxt(vectors_path)
        # Consistency check
        if self.embeddings.shape[0] != len(self.words_dictionary):
            raise ValueError(
                'SENNA embeddings and words files must have same number of rows'
                ' (%d vs. %d' %
                (self.embeddings.shape[0], len(self.words_dictionary)))
        logging.info('Done reading SENNA embeddings')


    def __getitem__(self, word):
        row = self.words_dictionary[word]
        return self.embeddings[row]
