from gflags import DEFINE_string, FLAGS, DuplicateFlagError
import logging
from nltk.tag.stanford import NERTagger
from os import path

from pipeline import Stage
from util import Enum

try:
    DEFINE_string('stanford_ser_path',
                  '/home/jesse/Documents/Work/Research/stanford-ner-2015-04-20',
                  'Path to Stanford NER directory')
    DEFINE_string(
        'stanford_ner_model_name', 'english.all.3class.distsim.crf.ser.gz',
        'Name of model file for Stanford NER')
except DuplicateFlagError as e:
    logging.warn('Ignoring redefinition of flag %s' % e.flagname)
    

class StanfordNERStage(Stage):
    NER_TYPES = Enum(['Person', 'Organization', 'Location', 'O'])

    def __init__(self, name):
        self.name = name
        # Omit models

    def train(self, instances):
        pass

    def test(self, instances):
        model_path = path.join(FLAGS.stanford_ser_path, 'classifiers',
                               FLAGS.stanford_ner_model_name)
        jar_path = path.join(FLAGS.stanford_ser_path, 'stanford-ner.jar')
        st = NERTagger(model_path, jar_path)
        tokens_by_sentence = [
            [token.original_text for token in sentence.tokens[1:]]
            for sentence in instances]
        # Batch process sentences (faster than repeatedly running Stanford NLP)
        ner_results = st.tag_sents(tokens_by_sentence)
        for sentence, sentence_result in zip(instances, ner_results):
            sentence.tokens[0].ner_tag = None # ROOT has no NER tag
            for token, token_result in zip(sentence.tokens[1:], sentence_result):
                tag = token_result[1]
                token.ner_tag = self.NER_TYPES.index(tag.title())
