from gflags import DEFINE_string, FLAGS, DuplicateFlagError
import threading
import logging
import pexpect
import time

from data import ParsedSentence
from util.metrics import ClassificationMetrics
from pipeline import Stage
from pipeline.models import Model
from stages import match_causation_pairs, print_instances_by_eval_result,\
    normalize_order

try:
    DEFINE_string('tregex_command',
                  '/home/jesse/Documents/Work/Research/'
                  'stanford-tregex-2014-10-26/tregex.sh',
                  'Command to run TRegex')
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
        self.tregex_processes = []

    @staticmethod
    def get_pattern_for_arg(connective, arg, arg_name):
        parent_sentence = connective.parent_sentence
        dep_path = parent_sentence.extract_dependency_path(connective, arg)
        pattern = '=connective'
        last_node = connective

        for source, target, dep_name in dep_path:
            forward_dependency = source is last_node
            if forward_dependency:
                next_node = target
            else:
                next_node = source

            if next_node is arg:
                node_name = '=' + arg_name
            else:
                node_name = ''

            if parent_sentence.is_clause_head(next_node):
                node_pos_pattern = '[<2 /^VB.*/ | < (__ <1 cop)]'
            else:
                node_pos_pattern = ('<2 /^%s.*/' % next_node.get_gen_pos())

            if forward_dependency:
                pattern = '%s < (__%s %s <1 %s' % (
                    pattern, node_name, node_pos_pattern, dep_name)
            else:
                pattern = '%s <1 %s > (__%s %s' % (
                    pattern, dep_name, node_name, node_pos_pattern)

            last_node = next_node

        pattern += ')' * len(dep_path)
        return pattern

    def _extract_patterns(self, sentences):
        # TODO: Extend this code to multiple-word connectives/args.
        # TODO: Figure out tree transformations to get rid of dumb things like
        # conjunctions that introduce spurious differences btw tregex_patterns?
        tregex_patterns = set()
        for sentence in sentences:
            for instance in sentence.causation_instances:
                if (len(instance.connective) == 1 and instance.cause is not None
                    and instance.effect is not None):
                    connective = instance.connective[0]
                    cause_head = sentence.get_head(instance.cause)
                    effect_head = sentence.get_head(instance.effect)

                    cause_pattern = (self.get_pattern_for_arg(
                        connective, cause_head, 'cause'))
                    effect_pattern = (self.get_pattern_for_arg(
                        connective, effect_head, 'effect'))
                    connective_pattern = (
                        '/^%s_[0-9]+$/=connective <2 /^%s.*/' % (
                            connective.lemma, connective.get_gen_pos()))
                    pattern = '%s : %s : %s' % (
                        connective_pattern, cause_pattern, effect_pattern)

                    #if pattern not in tregex_patterns:
                    #    logging.debug('Adding pattern:\n\t%s\n\tSentence: %s\n' % (
                    #        pattern, sentence.original_text))
                    tregex_patterns.add(pattern)

        self.tregex_patterns = list(tregex_patterns)

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
        # TODO: Somehow allow persisting the TRegex processes across test
        # batches? Perhaps by allowing framework to send an indication that a
        # particular batch is the last one?
        self._spawn_processes()

        # Interacting with the TRegex processes is heavily I/O-bound, so we use
        # threads to parallelize it a bit -- one thread per TRegex.

        # First, do sentence pre-processing that's common across all TRegexes.
        ptb_strings = []
        true_causation_pairs_by_sentence = []
        for sentence in sentences:
            sentence.possible_causations = []
            ptb_strings.append(sentence.to_ptb_tree_string())
            true_causation_pairs = [
                normalize_order(instance.get_cause_and_effect_heads())
                for instance in sentence.causation_instances]
            true_causation_pairs_by_sentence.append(set(
                [(arg_1.index, arg_2.index)
                 for arg_1, arg_2 in true_causation_pairs
                 if arg_1 is not None and arg_2 is not None]))

        # Define the function that each thread will perform: iterate over
        # sentences and modify them according to their results.
        class TregexProcessorThread(threading.Thread):
            def __init__(self, pattern, tregex_process, *args, **kwargs):
                super(TregexProcessorThread, self).__init__(*args, **kwargs)
                self.pattern = pattern
                self.tregex_process = tregex_process
                self.progress = 0

            def run(self):
                for sentence, ptb_string, true_causation_pairs in zip(
                    sentences, ptb_strings, true_causation_pairs_by_sentence):
                    # Interact with TRegex.
                    self.tregex_process.sendline(ptb_string)
                    # look for the double newline
                    self.tregex_process.expect("\r\n\r\n")
                    # skip tree num line
                    lines = self.tregex_process.before.split()[1:]

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

                    self.progress += 1

        # Start the threads.
        threads = []

        all_threads_done = False
        def report_progress_repeatedly():
            while(True):
                time.sleep(3)
                # This will be a slightly imprecise estimate, because progress
                # numbers are being grabbed in a non-threadsafe way. Whatever.
                progress = (sum([t.progress for t in threads])
                            / float(len(threads) * len(sentences)))
                if not all_threads_done:
                    logging.info("Tagging connectives: %1.0f%% complete"
                                 % (progress * 100))
                else:
                    break
        progress_reporter = threading.Thread(target=report_progress_repeatedly)
        progress_reporter.start()

        for pattern, tregex_process in zip(self.tregex_patterns,
                                           self.tregex_processes):
            new_thread = TregexProcessorThread(pattern, tregex_process)
            threads.append(new_thread)
            new_thread.start()
        for thread in threads:
            thread.join()
        all_threads_done = True

        self._kill_processes()
        logging.info("Done tagging possible connectives.")

    def _spawn_processes(self):
        for pattern in self.tregex_patterns:
            # To get rid of stderr junk, we need to redirect, which requires
            # spawning a shell.
            # TODO: see if we can redirect stderr to /dev/null in another way.
            tregex_process = pexpect.spawn(
                'sh',
                ['-c', '%s -u -s -o -l -N -h cause -h effect -filter "" "%s"'
                 ' 2> /dev/null' % (FLAGS.tregex_command, pattern)])
            tregex_process.delaybeforesend = 0
            tregex_process.setecho(False)
            self.tregex_processes.append(tregex_process)
        logging.debug("Spawned %d TRegex processes"
                      % len(self.tregex_patterns))


    def _kill_processes(self):
        for process in self.tregex_processes:
            process.kill(9)
        self.tregex_processes = []

class ConnectiveStage(Stage):
    def __init__(self, name):
        super(ConnectiveStage, self).__init__(
            name, [ConnectiveModel(part_type=ParsedSentence)])

    def get_produced_attributes(self):
        return ['possible_causations']

    def _extract_parts(self, sentence):
        return [sentence]

    def _begin_evaluation(self):
        self.tp, self.fn, self.fp = 0, 0, 0
        if FLAGS.sc_print_test_instances:
            self.tp_pairs, self.fn_pairs, self.fp_pairs = [], [], []

    def _evaluate(self, sentences):
        for sentence in sentences:
            predicted_pairs = [(pc.arg1, pc.arg2)
                               for pc in sentence.possible_causations]
            expected_pairs = [i.get_cause_and_effect_heads()
                              for i in sentence.causation_instances]
            tp, fn, fp = match_causation_pairs(
                expected_pairs, predicted_pairs, self.tp_pairs, self.fn_pairs,
                self.fp_pairs)

            self.tp += tp
            self.fn += fn
            self.fp += fp

    def _complete_evaluation(self):
        results = ClassificationMetrics(self.tp, self.fp, self.fn, None)
        if FLAGS.sc_print_test_instances:
            print_instances_by_eval_result(self.tp_pairs, self.fn_pairs, self.fp_pairs)
            self.tp_pairs, self.fn_pairs, self.fp_pairs = [], [], []
        return results
