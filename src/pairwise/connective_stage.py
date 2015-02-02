from gflags import DEFINE_string, DEFINE_bool, FLAGS, DuplicateFlagError, DEFINE_integer
import threading
import logging
from math import log10
import os
import Queue
import subprocess
import sys
import tempfile
import time

from data import ParsedSentence
from pipeline.models import Model
from pairwise import PairwiseCausalityStage
from util import pairwise
from util.metrics import ClassificationMetrics
from util.scipy import steiner_tree, longest_path_in_tree

try:
    DEFINE_string('tregex_dir',
                  '/home/jesse/Documents/Work/Research/'
                  'stanford-tregex-2014-10-26',
                  'Command to run TRegex')
    DEFINE_integer(
        'tregex_max_steiners', sys.maxsize,
        'Maximum number of Steiner nodes to be allowed in TRegex patterns')
    DEFINE_integer('tregex_max_threads', 30,
                   'Max number of TRegex processor threads')
    DEFINE_bool('tregex_print_patterns', False,
                'Whether to print all connective patterns')
    DEFINE_bool('tregex_print_test_instances', False,
                'Whether to print true positive, false positive, and false'
                ' negative instances after testing')

except DuplicateFlagError as e:
    logging.warn('Ignoring redefinition of flag %s' % e.flagname)


class PossibleCausation(object):
    def __init__(self, arg1, arg2, matching_pattern, correct):
        self.arg1 = arg1
        self.arg2 = arg2
        self.matching_pattern = matching_pattern
        self.correct = correct

class ConnectiveModel(Model):
    def __init__(self, *args, **kwargs):
        super(ConnectiveModel, self).__init__(*args, **kwargs)
        self.tregex_patterns = []

    @staticmethod
    def _get_connective_token_pattern(token, connective_index, node_names):
        node_name = 'connective_%d' % connective_index
        node_names[token.index] = node_name

        return (
            '/^%s_[0-9]+$/=%s <2 /^%s.*/' % (
            token.lemma, node_name, token.get_gen_pos()))

    @staticmethod
    def _get_non_connective_token_pattern(token, node_name, node_names):
        node_names[token.index] = node_name
        return '/.*_[0-9]+/=%s' % node_name
        '''
        parent_sentence = token.parent_sentence
        if parent_sentence.is_clause_head(token):
            pos_pattern = '[<2 /^VB.*/ | < (__ <1 cop)]'
        else:
            pos_pattern = ('<2 /^%s.*/' % token.get_gen_pos())

        return '/.*_[0-9+]/=%s %s' % (node_name, pos_pattern)
        '''

    @staticmethod
    def _get_steiner_pattern(token, steiner_index, node_names):
        return ConnectiveModel._get_non_connective_token_pattern(
            token, 'steiner_%d' % steiner_index, node_names)

    @staticmethod
    def _get_node_pattern(sentence, node, node_names, connective_nodes,
                          steiner_nodes, cause_head, effect_head):
        token = sentence.tokens[node]
        try:
            connective_index = connective_nodes.index(node)
            return ConnectiveModel._get_connective_token_pattern(
                token, connective_index, node_names)
        except ValueError: # It's not a connective node
            try:
                steiner_index = steiner_nodes.index(node)
                return ConnectiveModel._get_steiner_pattern(
                    token, steiner_index, node_names)
            except ValueError:  # It's an argument node
                node_name = ['cause', 'effect'][
                    token.index == effect_head.index]
                return ConnectiveModel._get_non_connective_token_pattern(
                    token, node_name, node_names)

    @staticmethod
    def _get_edge_pattern(edge_start, edge_end, sentence):
        edge_label = sentence.edge_labels[(edge_start, edge_end)]
        if edge_label in ['nsubj', 'csubj']:
            options = ['<1 nsubj', '<1 csubj']
        elif edge_label in ['nsubjpass', 'csubjpass']:
            options = ['<1 nsubjpass', '<1 csubjpass']
        else:
            options = ['<1 ' + edge_label]
        if edge_label != 'dep':
            options += ['<1 dep']
        return '[%s]' % ' | '.join(options)

    @staticmethod
    def _get_pattern_for_instance(sentence, connective, cause_head,
                                  effect_head):
        connective_nodes = [token.index for token in connective]
        required_token_indices = list(set(# Eliminate potential duplicates
            [cause_head.index, effect_head.index] + connective_nodes))

        steiner_nodes, steiner_graph = steiner_tree(
            sentence.edge_graph, required_token_indices,
            sentence.path_costs, sentence.path_predecessors)

        if len(steiner_nodes) > FLAGS.tregex_max_steiners:
            logging.debug(
                "Ignoring potentially very long pattern (sentence: %s)"
                % sentence.original_text)
            return (None, None)

        # To generate the pattern, we start by generating one long string that
        # can be checked easily by TRegex. That'll be the biggest chunk of the
        # pattern. It consists of the longest path through the Steiner tree
        # edges.

        # Start the longest path search from a node we know is actually in the
        # tree we're looking for.
        longest_path = longest_path_in_tree(steiner_graph, connective_nodes[0])
        pattern = ''
        node_names = {}
        edges = [(None, longest_path[0])] + list(pairwise(longest_path))
        for edge_start, edge_end in edges:
            end_pattern = ConnectiveModel._get_node_pattern(
                sentence, edge_end, node_names, connective_nodes,
                steiner_nodes, cause_head, effect_head)
            if edge_start is not None:
                if steiner_graph[edge_start, edge_end]: # forward edge
                    edge_pattern = ConnectiveModel._get_edge_pattern(
                        edge_start, edge_end, sentence)
                    pattern = '%s < (%s %s' % (pattern, end_pattern,
                                               edge_pattern)
                else: # back edge
                    edge_pattern = ConnectiveModel._get_edge_pattern(
                        edge_end, edge_start, sentence)
                    pattern = '%s %s > (%s' % (pattern, edge_pattern,
                                               end_pattern)

            else:
                pattern = '%s(%s' % (pattern, end_pattern)
        pattern += ')' * len(edges)

        # Next, we need to make sure all the edges that weren't included in the
        # longest path get incorporated into the pattern. For this, it's OK to
        # have a few colon-separated pattern segments.
        def get_named_node_pattern(node):
            try:
                return '=' + node_names[node]
            except KeyError:  # Node hasn't been named and given a pattern yet
                return '(%s)' % ConnectiveModel._get_node_pattern(
                    sentence, node, node_names, connective_nodes,
                    steiner_nodes, cause_head, effect_head)

        for edge_start, edge_end in zip(*steiner_graph.nonzero()):
            if ((edge_start, edge_end) in edges
                or (edge_end, edge_start) in edges):
                continue
            start_pattern = get_named_node_pattern(edge_start)
            end_pattern = get_named_node_pattern(edge_end)
            edge_pattern = ConnectiveModel._get_edge_pattern(
                edge_start, edge_end, sentence)
            pattern = '%s : (%s < (%s %s))' % (pattern, start_pattern,
                                               end_pattern, edge_pattern)

        node_names_to_print = [name for name in node_names.values()
                               if name not in ['cause', 'effect']]
        return pattern, node_names_to_print

    def _extract_patterns(self, sentences):
        # TODO: Extend this to work with cases of missing arguments.
        # TODO: Figure out tree transformations to get rid of dumb things like
        # conjunctions that introduce spurious differences btw patterns?
        self.tregex_patterns = []
        patterns_seen = set()
        if FLAGS.tregex_print_patterns:
            print 'Patterns:'
        for sentence in sentences:
            for instance in sentence.causation_instances:
                if instance.cause != None and instance.effect is not None:
                    connective = instance.connective
                    cause_head = sentence.get_head(instance.cause)
                    effect_head = sentence.get_head(instance.effect)
                    pattern, node_names = self._get_pattern_for_instance(
                        sentence, connective, cause_head, effect_head)

                    if pattern is None:
                        continue

                    if pattern not in patterns_seen:
                        if FLAGS.tregex_print_patterns:
                            print pattern
                            print 'Sentence:', sentence.original_text
                            print
                        patterns_seen.add(pattern)
                        connective_lemmas = [t.lemma for t in connective]
                        self.tregex_patterns.append((pattern, node_names,
                                                     connective_lemmas))

    class TregexProcessorThread(threading.Thread):
        def __init__(self, sentences, ptb_strings, queue,
                     true_causation_pairs_by_sentence, *args, **kwargs):
            super(ConnectiveModel.TregexProcessorThread, self).__init__(
                *args, **kwargs)
            self.sentences = sentences
            self.ptb_strings = ptb_strings
            self.true_causation_pairs_by_sentence = (
                true_causation_pairs_by_sentence)
            self.queue = queue
            self.output_file = None
            self.total_bytes_output = 0

        dev_null = open('/dev/null', 'w')

        def run(self):
            try:
                while(True):
                    (pattern, node_labels, possible_sentence_indices) = (
                        self.queue.get_nowait())
                    if not possible_sentence_indices: # no sentences to scan
                        self.queue.task_done()
                        continue

                    possible_trees = [self.ptb_strings[i]
                                      for i in possible_sentence_indices]
                    possible_sentences = [self.sentences[i]
                                          for i in possible_sentence_indices]
                    possible_true_causation_pairs = [
                        self.true_causation_pairs_by_sentence[i]
                        for i in possible_sentence_indices]

                    with tempfile.NamedTemporaryFile('w') as tree_file:
                        tree_file.writelines(possible_trees)
                        # Make sure the file is synced for threads to access
                        tree_file.flush()
                        self._process_pattern(
                            pattern, possible_sentences, tree_file.name,
                            possible_true_causation_pairs)
                        self.queue.task_done()
            except Queue.Empty: # no more items in queue
                return

        def _process_pattern(self, pattern, possible_sentences, tree_file_path,
                             possible_true_causation_pairs):
            # Create output file
            with tempfile.NamedTemporaryFile('w+b') as self.output_file:
                #print "Processing", pattern, "to", tregex_output.name
                # TODO: Make this use the node labels to also retrieve the
                # possible connective tokens.
                tregex_args = '-u -s -o -l -N -h cause -h effect'.split()
                tregex_command = (
                    [os.path.join(FLAGS.tregex_dir, 'tregex.sh')] + tregex_args
                    + [pattern, tree_file_path])
                subprocess.call(tregex_command, stdout=self.output_file,
                                stderr=self.dev_null)
                self.output_file.seek(0)

                # For each sentence, we leave the file positioned at the next
                # tree number line.
                for sentence, true_causation_pairs in zip(
                    possible_sentences, possible_true_causation_pairs):
                    # Read TRegex output for the sentence.
                    self.output_file.readline() # skip tree num line
                    next_line = self.output_file.readline().strip()
                    lines = []
                    while next_line:
                        lines.append(next_line)
                        next_line = self.output_file.readline().strip()

                    # Parse TRegex output.
                    line_pairs = zip(lines[0::2], lines[1::2])
                    for line_pair in line_pairs:
                        index_pair = [int(line.split("_")[-1])
                                      for line in line_pair]
                        index_pair = tuple(sorted(index_pair))

                        # Mark sentence if possible connective is present.
                        t1_index, t2_index = index_pair
                        in_gold = index_pair in true_causation_pairs
                        possible = PossibleCausation(
                            sentence.tokens[t1_index],
                            sentence.tokens[t2_index], pattern, in_gold)

                        # THIS IS THE ONLY LINE THAT MUTATES SHARED DATA.
                        # It is thread-safe, because lists are thread-safe, and
                        # we never reassign sentence.possible_causations.
                        sentence.possible_causations.append(possible)

                # Tell the progress reporter how far we've gotten, so that it
                # will know progress for patterns that have already finished.
                self.total_bytes_output += self.output_file.tell()

            self.output_file = None

        def get_progress(self):
            try:
                return self.total_bytes_output + self.output_file.tell()
            except (AttributeError, IOError, ValueError):
                # AttributeError indicates that self.output_file was None.
                # IOError/ValueError indicate that we managed to ask for file
                # size just after the file was closed. Either way, that means
                # that now the total number of bytes has been recorded.
                return self.total_bytes_output

    def train(self, sentences):
        self._extract_patterns(sentences)
        # Now that we have all the patterns, we also need to make sure all the
        # instances we pass along to the next stage have input matching what
        # will be passed along at test time. That means we need false negatives
        # in exactly the same places that they'll be at test time, so we just
        # run test() to find all the correct and spurious matches.
        logging.debug("Running test to generate input for next stage")
        self.test(sentences)


    @staticmethod
    def _preprocess_sentences(sentences):
        ptb_strings = []
        true_causation_pairs_by_sentence = []
        for sentence in sentences:
            sentence.possible_causations = []
            ptb_strings.append(sentence.to_ptb_tree_string() + '\n')
            true_causation_pairs = [
                ConnectiveStage.normalize_order(
                    instance.get_cause_and_effect_heads())
                for instance in sentence.causation_instances]
            true_causation_pairs_by_sentence.append(
                set([(arg_1.index, arg_2.index)
                     for (arg_1, arg_2) in true_causation_pairs
                     if arg_1 is not None and arg_2 is not None]))
            
        with tempfile.NamedTemporaryFile('w', delete=False) as tree_file:
            tree_file.writelines(ptb_strings)
            tree_file.flush()
            with tempfile.NamedTemporaryFile('w+b', delete=False) as surgeried_file:
                tsurgeon_command = (
                    ([os.path.join(FLAGS.tregex_dir, 'tsurgeon.sh'), '-s',
                      '-treeFile', tree_file.name] +
                     [os.path.join('pairwise', tr) for tr in [
                      'normalize_passives.ts', 'normalize_vmod_passives.ts',
                      'postprocess_vmod_passives.ts']]))
                subprocess.call(tsurgeon_command, stdout=surgeried_file)
                surgeried_file.seek(0)
                ptb_strings = surgeried_file.readlines()

        return ptb_strings, true_causation_pairs_by_sentence

    @staticmethod
    def _make_progress_reporter(threads, total_estimated_bytes,
                                    all_threads_done):
        def report_progress_loop():
            while(True):
                time.sleep(4)
                if all_threads_done[0]:
                    return
                bytes_output = sum([t.get_progress() for t in threads])
                # Never allow > 99% completion as long as we're still running.
                # (This could theoretically happen if our estimated max sizes
                # turned out to be off.)
                try:
                    progress = min(
                        bytes_output / float(total_estimated_bytes), 0.99)
                except ZeroDivisionError:
                    progress = 0
                if not all_threads_done[0]: # Make sure we're still going
                    logging.info("Tagging connectives: %1.0f%% complete"
                                 % (progress * 100))
                else:
                    break

        progress_reporter = threading.Thread(target=report_progress_loop)
        progress_reporter.daemon = True
        return progress_reporter

    @staticmethod
    def _filter_sentences_for_pattern(sentences, pattern, connective_lemmas):
        possible_sentence_indices = []
        for i, sentence in enumerate(sentences):
            token_lemmas = [token.lemma for token in sentence.tokens]
            # TODO: Should we filter here by whether there are enough tokens in
            # the sentence to match the rest of the pattern, too?
            if all([connective_lemma in token_lemmas
                    for connective_lemma in connective_lemmas]):
                possible_sentence_indices.append(i)

        return possible_sentence_indices

    def test(self, sentences):
        logging.info('Tagging possible connectives...')
        start_time = time.time()

        ptb_strings, true_causation_pairs_by_sentence = (
            self._preprocess_sentences(sentences))

        # Interacting with the TRegex processes is heavily I/O-bound, plus we
        # really want multiple TRegex processes running in parallel, so we farm
        # out patterns to worker threads.

        # Queue up the patterns
        total_estimated_bytes = 0
        queue = Queue.Queue()
        for pattern, node_labels, connective_lemmas in self.tregex_patterns:
            possible_sentence_indices = self._filter_sentences_for_pattern(
                sentences, pattern, connective_lemmas)
            queue.put_nowait((pattern, node_labels,
                              possible_sentence_indices))
            # Estimate total output file size for this pattern: each
            # sentence has sentence #, + 3 bytes for : and newlines.
            # As a very rough estimate, matching node names increase the
            # bytes used by ~1.85x.
            total_estimated_bytes += sum(
                1.85 * (int(log10(i + 1)) + 3)
                for i in range(len(possible_sentence_indices)))

        # Start the threads
        threads = []
        for _ in range(FLAGS.tregex_max_threads):
            new_thread = self.TregexProcessorThread(
                sentences, ptb_strings, queue,
                true_causation_pairs_by_sentence)
            threads.append(new_thread)
            new_thread.start()

        # Set up progress reporter and wait for threads to finish.
        all_threads_done = [False] # use a list for passing by ref
        progress_reporter = self._make_progress_reporter(
            threads, total_estimated_bytes, all_threads_done)
        try:
            progress_reporter.start()
            queue.join()
        finally:
            # Make sure progress reporter exits
            all_threads_done[0] = True

        elapsed_seconds = time.time() - start_time
        logging.info("Done tagging possible connectives in %0.2f seconds"
                     % elapsed_seconds)

class ConnectiveStage(PairwiseCausalityStage):
    def __init__(self, name):
        super(ConnectiveStage, self).__init__(
            print_test_instances=FLAGS.tregex_print_test_instances, name=name,
            models=[ConnectiveModel(part_type=ParsedSentence)])

    def get_produced_attributes(self):
        return ['possible_causations']

    @staticmethod
    def average_eval_pairs(metrics_pairs):
        return (ClassificationMetrics.average([m1 for m1, _ in metrics_pairs]),
                ClassificationMetrics.average([m2 for _, m2 in metrics_pairs]))

    def _extract_parts(self, sentence):
        return [sentence]

    def _begin_evaluation(self):
        PairwiseCausalityStage._begin_evaluation(self)
        self.pairwise_only_metrics = ClassificationMetrics()

    def _complete_evaluation(self):
        all_instances_metrics = PairwiseCausalityStage._complete_evaluation(
            self)
        pairwise_only_metrics = self.pairwise_only_metrics
        del self.pairwise_only_metrics
        return (all_instances_metrics, pairwise_only_metrics)

    def _evaluate(self, sentences):
        for sentence in sentences:
            predicted_pairs = [(pc.arg1, pc.arg2)
                               for pc in sentence.possible_causations]
            expected_pairs = [i.get_cause_and_effect_heads()
                              for i in sentence.causation_instances]
            tp, fp, fn = self.match_causation_pairs(
                expected_pairs, predicted_pairs, self.tp_pairs, self.fp_pairs,
                self.fn_pairs, self.all_instances_metrics)

            self.pairwise_only_metrics.tp += tp
            self.pairwise_only_metrics.fp += fp
            fns_to_ignore = sum(1 for pair in expected_pairs if None in pair)
            self.pairwise_only_metrics.fn += fn - fns_to_ignore
