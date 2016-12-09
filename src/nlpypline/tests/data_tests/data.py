from __future__ import absolute_import

import unittest

from nlpypline.data import StanfordParsedSentence
from nlpypline.data.io import StanfordParsedSentenceReader
from tests import get_sentences_from_file


class HeadFindingTests(unittest.TestCase):
    def setUp(self):
        self.sentences = get_sentences_from_file(
            StanfordParsedSentenceReader, 'DataTest', 'data_test.txt')

    def _check_head(self, tokens, text, correct_head_index, correct_head_text):
        # Sanity check: make sure we grabbed the intended tokens.
        original_tokens_text = ' '.join([t.original_text for t in tokens])
        self.assertEqual(original_tokens_text, text)

        sentence = tokens[0].parent_sentence
        correct_head = sentence.tokens[correct_head_index]
        # Sanity check: make sure supposedly correct head was the intended one.
        self.assertEqual(correct_head.original_text, correct_head_text)

        head = sentence.get_head(tokens)
        self.assertEqual(correct_head, head)

    def testPhraseHeadFinding(self):
        sentence = self.sentences[0]
        self._check_head(sentence.tokens[10:17],
                         'we have not yet found sufficient replacement',
                         14, 'found')

        sentence = self.sentences[1]
        self._check_head(
            sentence.tokens[5:10], 'a market sensitive regulatory authority',
            9, 'authority')
        self._check_head(sentence.tokens[23:-1],
                         'we have investors now who are unwilling to invest'
                         ' even in things they should',
                         24, 'have')

    def testFragmentedHeadFinding(self):
        sentence = self.sentences[0]
        # Check that with actual fragments that are at equal levels, verbs are
        # preferred.
        self._check_head(
            sentence.tokens[34:36] + [sentence.tokens[40]],
            'the borrower repay',
            40, 'repay')

        # Likewise for preferring copulas.
        sentence = self.sentences[2]
        self._check_head(
            sentence.tokens[3:8], "him it was n't bad", 7, 'bad')

        sentence = self.sentences[0]
        # "sufficient replacement for..." got parsed wrong, so a phrase like
        # "sufficient replacement for the discipline..." will be seen as
        # fragmented. Between 'replacement' and 'lending', we should choose the
        # verb.
        self._check_head(
            sentence.tokens[15:28],
            'sufficient replacement for the discipline of a lender not lending'
            ' to a borrower',
            24, 'lending')

    def testXcompWithSubjectHeadFinding(self):
        sentence = self.sentences[3]
        self._check_head(
            [sentence.tokens[1]] + sentence.tokens[6:10],
            'I to find my daughter', 7, 'find')
        # Check in the other order, just to make sure that it's not dependent on
        # the order in which the tokens are considered. (Since this test
        # requires the head finder to consider things other than tree depth,
        # this is possible if checks for better head qualifications aren't done
        # symmetrically.)
        self._check_head(
            sentence.tokens[6:10] + [sentence.tokens[1]],
            'to find my daughter I', 7, 'find')

    def testHeadDuplicatedAsChildsArg(self):
        sentence = self.sentences[4]
        # Duplicate passive subject.
        sentence.edge_graph[5, 4] = 1.0
        sentence.edge_labels[(5, 4)] = 'dobj'
        sentence.edge_labels[(5, 8)] = 'nsubj'
        self._check_head(
            sentence.tokens[4:9],
            'injuries caused by an explosion', 4, 'injuries')

class DependencyPathTests(unittest.TestCase):
    def setUp(self):
        self.sentences = get_sentences_from_file(
            StanfordParsedSentenceReader, 'DataTest', 'data_test.txt')

    def testAllForwardPaths(self):
        sentence = self.sentences[0]
        path_14_9 = sentence.extract_dependency_path(sentence.tokens[14],
                                                     sentence.tokens[9])
        self.assertEqual('mark', str(path_14_9))
        path_8_9 = sentence.extract_dependency_path(sentence.tokens[8],
                                                     sentence.tokens[9])
        self.assertEqual('ccomp mark', str(path_8_9))

    def testAllBackwardPaths(self):
        sentence = self.sentences[0]
        path_9_14 = sentence.extract_dependency_path(sentence.tokens[9],
                                                     sentence.tokens[14])
        self.assertEqual("mark'", str(path_9_14))
        path_9_8 = sentence.extract_dependency_path(sentence.tokens[9],
                                                     sentence.tokens[8])
        self.assertEqual("mark' ccomp'", str(path_9_8))

    def testMixedPaths(self):
        sentence = self.sentences[0]
        path_27_32 = sentence.extract_dependency_path(sentence.tokens[27],
                                                     sentence.tokens[32])
        self.assertEqual("nmod' advcl", str(path_27_32))
        path_17_15 = sentence.extract_dependency_path(sentence.tokens[17],
                                                     sentence.tokens[15])
        self.assertEqual("mark' advcl' dobj amod", str(path_17_15))
        path_15_17 = sentence.extract_dependency_path(sentence.tokens[15],
                                                     sentence.tokens[17])
        self.assertEqual("amod' dobj' advcl mark", str(path_15_17))

    def testDomination(self):
        sentence = self.sentences[0]

        # Test single links
        domination = sentence.get_domination_relation(sentence.tokens[14],
                                                      sentence.tokens[13])
        self.assertEqual(
            domination, StanfordParsedSentence.DOMINATION_DIRECTION.Dominates)

        domination = sentence.get_domination_relation(sentence.tokens[13],
                                                      sentence.tokens[14])
        self.assertEqual(
            domination, StanfordParsedSentence.DOMINATION_DIRECTION.DominatedBy)

        # Test multiple links
        domination = sentence.get_domination_relation(sentence.tokens[14],
                                                      sentence.tokens[26])
        self.assertEqual(domination,
                         StanfordParsedSentence.DOMINATION_DIRECTION.Dominates)

        domination = sentence.get_domination_relation(sentence.tokens[26],
                                                      sentence.tokens[14])
        self.assertEqual(
            domination, StanfordParsedSentence.DOMINATION_DIRECTION.DominatedBy)

        # Test unrelated
        domination = sentence.get_domination_relation(sentence.tokens[4],
                                                      sentence.tokens[14])
        self.assertEqual(
            domination,
            StanfordParsedSentence.DOMINATION_DIRECTION.Independent)
