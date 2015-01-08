from gflags import DEFINE_string, FLAGS, DuplicateFlagError, DEFINE_integer
import threading
import logging
from math import log10
import Queue
import subprocess
import tempfile
import time

from data import ParsedSentence
from util.metrics import ClassificationMetrics
from pipeline import Stage
from pipeline.models import Model
from stages import match_causation_pairs, print_instances_by_eval_result, normalize_order
from util.scipy import steiner_tree

try:
    DEFINE_string('tregex_command',
                  '/home/jesse/Documents/Work/Research/'
                  'stanford-tregex-2014-10-26/tregex.sh',
                  'Command to run TRegex')
    DEFINE_integer(
        'tregex_max_steiners', 4,
        'Maximum number of Steiner nodes to be allowed in TRegex patterns')
    DEFINE_integer('tregex_max_threads', 50,
                   'Max number of TRegex processor threads')
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
            '(/^%s_[0-9]+$/=%s <2 /^%s.*/)' % (
            token.lemma, node_name, token.get_gen_pos()))

    @staticmethod
    def _get_non_connective_token_pattern(token, node_name, node_names):
        node_names[token.index] = node_name

        parent_sentence = token.parent_sentence
        if parent_sentence.is_clause_head(token):
            pos_pattern = '[<2 /^VB.*/ | < (__ <1 cop)]'
        else:
            pos_pattern = ('<2 /^%s.*/' % token.get_gen_pos())

        return '(__=%s %s)' % (node_name, pos_pattern)

    @staticmethod
    def _get_steiner_pattern(token, steiner_index, node_names):
        return ConnectiveModel._get_non_connective_token_pattern(
            token, 'steiner_%d' % steiner_index, node_names)

    @staticmethod
    def _get_pattern_for_instance(sentence, connective, cause_head, effect_head):
        node_names = {}
        sub_patterns = [
            ConnectiveModel._get_connective_token_pattern(token, i,
                                                          node_names)
            for i, token in enumerate(connective)]

        for token, node_name in zip([cause_head, effect_head],
                                    ['cause', 'effect']):
            sub_patterns.append(
                ConnectiveModel._get_non_connective_token_pattern(
                    token, node_name, node_names))

        required_token_indices = list(set( # Eliminate potential duplicates
            [cause_head.index, effect_head.index]
            + [token.index for token in connective]))
        steiner_nodes, steiner_graph = steiner_tree(
            sentence.edge_graph, required_token_indices,
            sentence.path_costs, sentence.path_predecessors)

        if len(steiner_nodes) > FLAGS.tregex_max_steiners:
            logging.debug(
                "Ignoring potentially very long pattern (sentence: %s)"
                % sentence.original_text)
            return (None, None)

        for i, token_index in enumerate(steiner_nodes):
            sub_patterns.append(ConnectiveModel._get_steiner_pattern(
                sentence.tokens[token_index], i, node_names))

        for edge_start, edge_end in zip(*steiner_graph.nonzero()):
            start_name, end_name = [node_names[index]
                                    for index in [edge_start, edge_end]]
            edge_label = sentence.edge_labels[(edge_start, edge_end)]
            edge_pattern = '(=%s < (=%s <1 %s))' % (start_name, end_name,
                                                    edge_label)
            sub_patterns.append(edge_pattern)

        node_names_to_print = tuple(v for v in node_names.values()
                                    if not v.startswith('steiner'))
        return (' : '.join(sub_patterns), node_names_to_print)

    def _extract_patterns(self, sentences):
        # TODO: Extend this to work with cases of missing arguments.
        # TODO: Figure out tree transformations to get rid of dumb things like
        # conjunctions that introduce spurious differences btw patterns?
        self.tregex_patterns = []
        patterns_seen = set()
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
                        #logging.debug(
                        #    'Adding pattern:\n\t%s\n\tSentence: %s\n'
                        #    % (pattern, sentence.original_text))
                        patterns_seen.add(pattern)
                        self.tregex_patterns.append((pattern, node_names))

    class TregexProcessorThread(threading.Thread):
        def __init__(self, trees_file_path, sentences, queue,
                     true_causation_pairs_by_sentence, *args, **kwargs):
            super(ConnectiveModel.TregexProcessorThread, self).__init__(
                *args, **kwargs)
            self.trees_file_path = trees_file_path
            self.sentences = sentences
            self.true_causation_pairs_by_sentence = (
                true_causation_pairs_by_sentence)
            self.queue = queue
            self.output_file = None
            self.total_bytes_output = 0

        dev_null = open('/dev/null', 'w')

        def run(self):
            try:
                while(True):
                    pattern = self.queue.get_nowait()
                    self._process_pattern(pattern)
                    self.queue.task_done()
            except Queue.Empty: # no more items in queue
                return

        def _process_pattern(self, pattern):
            # Create output file
            with tempfile.NamedTemporaryFile('w+b') as self.output_file:
                #print "Processing", pattern, "to", tregex_output.name
                # TODO: Make this use the node labels to also retrieve the
                # possible connective tokens.
                tregex_args = '-u -s -o -l -N -h cause -h effect'.split()
                full_tregex_command = (
                    [FLAGS.tregex_command] + tregex_args
                    + [pattern, self.trees_file_path])
                subprocess.call(full_tregex_command, stdout=self.output_file,
                                stderr=self.dev_null)
                self.output_file.seek(0)

                # For each sentence, we leave the file positioned at the next
                # tree number line.
                for sentence, true_causation_pairs in zip(
                    self.sentences, self.true_causation_pairs_by_sentence):
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

                self.total_bytes_output += self.output_file.tell()
                self.output_file = None


    def train(self, sentences):
        self._extract_patterns(sentences)
        # Now that we have all the patterns, we also need to make sure all the
        # instances we pass along to the next stage have input matching what
        # will be passed along at test time. That means we need false negatives
        # in exactly the same places that they'll be at test time, so we just
        # run test() to find all the correct and spurious matches.
        logging.debug("Running test to generate input for next stage")
        self.test(sentences)

    def test(self, sentences):
        logging.info('Tagging possible connectives...')
        start_time = time.time()
        # Interacting with the TRegex processes is heavily I/O-bound, so we use
        # threads to parallelize it a bit -- one thread per TRegex.

        # First, do sentence pre-processing that's common across all TRegexes.
        ptb_strings = []
        true_causation_pairs_by_sentence = []
        for sentence in sentences:
            sentence.possible_causations = []
            # Add newlines for writing to file later.
            ptb_strings.append(sentence.to_ptb_tree_string() + '\n')
            true_causation_pairs = [
                normalize_order(instance.get_cause_and_effect_heads())
                for instance in sentence.causation_instances]
            true_causation_pairs_by_sentence.append(set(
                [(arg_1.index, arg_2.index)
                 for arg_1, arg_2 in true_causation_pairs
                 if arg_1 is not None and arg_2 is not None]))

        # Set up progress reporter
        threads = []
        all_threads_done = False
        def report_progress_loop():
            # Start with a size estimate of each file where its max size is
            # 1.3x the size of a file where no sentences have matched.
            estimated_sentence_sizes = [1.3 * (int(log10(i+1)) + 3)
                                        for i in range(len(sentences))]
            total_bytes = (len(self.tregex_patterns) *
                           sum(estimated_sentence_sizes))
            while(True):
                time.sleep(4)
                bytes_output = sum([t.total_bytes_output + t.output_file.tell()
                                    if t.output_file else t.total_bytes_output
                                    for t in threads])
                # Never allow > 99% completion as long as we're still running.
                # (This could theoretically happen if our estimated max sizes
                # turned out to be way off.)
                progress = min(bytes_output / float(total_bytes), 0.99)

                if not all_threads_done:
                    logging.info("Tagging connectives: %1.0f%% complete"
                                 % (progress * 100))
                else:
                    break
        progress_reporter = threading.Thread(target=report_progress_loop)
        progress_reporter.daemon = True

        try:
            with tempfile.NamedTemporaryFile('w') as trees_file:
                trees_file.writelines(ptb_strings)
                # Make sure the file is synced for threads to access
                trees_file.flush()

                progress_reporter.start()

                # Queue up the patterns and start the threads chewing on them.
                queue = Queue.Queue()
                for pattern, node_labels in self.tregex_patterns:
                    queue.put_nowait(pattern)
                for _ in range(FLAGS.tregex_max_threads):
                    new_thread = self.TregexProcessorThread(
                        trees_file.name, sentences, queue,
                        true_causation_pairs_by_sentence)
                    threads.append(new_thread)
                    new_thread.start()
                queue.join()
        finally:
            # Make sure progress reporter exits
            all_threads_done = True

        elapsed_seconds = time.time() - start_time
        logging.info("Done tagging possible connectives in %0.2f seconds"
                     % elapsed_seconds)

class ConnectiveStage(Stage):
    def __init__(self, name):
        super(ConnectiveStage, self).__init__(
            name, [ConnectiveModel(part_type=ParsedSentence)])

    def get_produced_attributes(self):
        return ['possible_causations']

    def _extract_parts(self, sentence):
        return [sentence]

    def _begin_evaluation(self):
        self.tp, self.fp, self.fn = 0, 0, 0
        self.tp_pairs, self.fp_pairs, self.fn_pairs = [], [], []
        print self.tp_pairs

    def _evaluate(self, sentences):
        for sentence in sentences:
            predicted_pairs = [(pc.arg1, pc.arg2)
                               for pc in sentence.possible_causations]
            expected_pairs = [i.get_cause_and_effect_heads()
                              for i in sentence.causation_instances]
            tp, fp, fn = match_causation_pairs(
                expected_pairs, predicted_pairs, self.tp_pairs, self.fp_pairs,
                self.fn_pairs)

            self.tp += tp
            self.fp += fp
            self.fn += fn

    def _complete_evaluation(self):
        results = ClassificationMetrics(self.tp, self.fp, self.fn, None)
        if FLAGS.sc_print_test_instances:
            print_instances_by_eval_result(self.tp_pairs, self.fp_pairs,
                                           self.fn_pairs)
            self.tp_pairs, self.fp_pairs, self.fn_pairs = [], [], []
        return results
