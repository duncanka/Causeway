from gflags import DEFINE_string, DEFINE_bool, FLAGS, DuplicateFlagError, \
    DEFINE_integer, DEFINE_enum
import threading
import logging
from math import log10
import os
import Queue
import subprocess
import tempfile
import time

from data import ParsedSentence, Token
from pipeline.models import Model
from causality_pipelines.pairwise import PairwiseCausalityStage
from util import pairwise
from util.metrics import ClassificationMetrics
from util.nltk import subtree_at_index, index_of_subtree
from util.scipy import steiner_tree, longest_path_in_tree

try:
    DEFINE_string('tregex_dir',
                  '/home/jesse/Documents/Work/Research/'
                  'stanford-tregex-2014-10-26',
                  'Command to run TRegex')
    DEFINE_integer(
        'tregex_max_steiners', 10,
        'Maximum number of Steiner nodes to be allowed in TRegex patterns')
    DEFINE_integer('tregex_max_threads', 30,
                   'Max number of TRegex processor threads')
    DEFINE_bool('tregex_print_patterns', False,
                'Whether to print all connective patterns')
    DEFINE_bool('tregex_print_test_instances', False,
                'Whether to print true positive, false positive, and false'
                ' negative instances after testing')
    DEFINE_enum('tregex_pattern_type', 'dependency',
                ['dependency', 'constituency'],
                'Type of tree to generate and run TRegex patterns with')

except DuplicateFlagError as e:
    logging.warn('Ignoring redefinition of flag %s' % e.flagname)


class PossibleCausation(object):
    def __init__(self, arg1, arg2, matching_pattern, correct):
        self.arg1 = arg1
        self.arg2 = arg2
        self.matching_pattern = matching_pattern
        self.correct = correct

class TRegexConnectiveModel(Model):
    def __init__(self, *args, **kwargs):
        super(TRegexConnectiveModel, self).__init__(*args, **kwargs)
        self.tregex_patterns = []

    def train(self, sentences):
        ptb_strings, true_causation_pairs_by_sentence = (
            self._extract_patterns(sentences))
        # Now that we have all the patterns, we also need to make sure all the
        # instances we pass along to the next stage have input matching what
        # will be passed along at test time. That means we need false negatives
        # in exactly the same places that they'll be at test time, so we just
        # run test() to find all the correct and spurious matches.
        logging.debug("Running test to generate input for next stage")
        self.test(sentences, ptb_strings, true_causation_pairs_by_sentence)

    def test(self, sentences, ptb_strings=None,
             true_causation_pairs_by_sentence=None):
        logging.info('Tagging possible connectives...')
        start_time = time.time()

        if ptb_strings is None or true_causation_pairs_by_sentence is None:
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
        all_threads_done = [False] # use a list for passing by ref (EVIL HACK)
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

    #####################################
    # Sentence preprocessing
    #####################################

    @staticmethod
    def _preprocess_sentences(sentences):
        logging.info("Preprocessing sentences...")
        ptb_strings = []
        true_causation_pairs_by_sentence = []
        for sentence in sentences:
            sentence.possible_causations = []
            if FLAGS.tregex_pattern_type == 'dependency':
                ptb_strings.append(sentence.dep_to_ptb_tree_string() + '\n')
            else:
                ptb_strings.append(sentence.constituency_tree.pformat() + '\n')
            true_causation_pairs = [
                TRegexConnectiveStage.normalize_order(
                    instance.get_cause_and_effect_heads())
                for instance in sentence.causation_instances]
            true_causation_pairs_by_sentence.append(
                set([(arg_1.index, arg_2.index)
                     for (arg_1, arg_2) in true_causation_pairs
                     if arg_1 is not None and arg_2 is not None]))

        if FLAGS.tregex_pattern_type == 'dependency':
            # Order matters a lot here.
            tsurgeon_script_names = [
                'normalize_passives',
                'normalize_vmod_passives_1',
                'normalize_vmod_passives_2',
                'normalize_vmod_no_agent_1',
                'normalize_vmod_no_agent_2',
                'normalize_vmod_no_agent_3']
            tsurgeon_script_names = [
                os.path.join('causality_pipelines', 'pairwise', 'tsurgeon_dep',
                             script_name) + '.ts'
                for script_name in tsurgeon_script_names]

            with tempfile.NamedTemporaryFile('w', delete=False) as tree_file:
                encoded_strings = [s.encode('utf-8') for s in ptb_strings]
                tree_file.writelines(encoded_strings)
                tree_file.flush()
                with tempfile.NamedTemporaryFile(
                    'w+b', delete=False) as surgeried_file:
                    tsurgeon_command = (
                        ([os.path.join(FLAGS.tregex_dir, 'tsurgeon.sh'), '-s',
                          '-treeFile', tree_file.name]
                         + tsurgeon_script_names))
                    subprocess.call(tsurgeon_command, stdout=surgeried_file)
                    surgeried_file.seek(0)
                    ptb_strings = surgeried_file.readlines()
        else:
            # Temporary measure until we get TSurgeon scripts updated for
            # constituency parses: don't do any real preprocessing.
            # TODO: Implement constituency scripts, and move TSurgeon-running
            # code to be shared.
            pass

        logging.info('Done preprocessing.')
        return ptb_strings, true_causation_pairs_by_sentence

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

    #####################################
    # Pattern generation
    #####################################

    @staticmethod
    def _get_dep_node_pattern(
        sentence, node_index, node_names, connective_indices,
        steiner_nodes, cause_head, effect_head):
        def non_connective_pattern(node_name):
            node_names[node_index] = node_name
            return '/.*_[0-9]+/=%s' % node_name
            '''
            parent_sentence = token.parent_sentence
            if parent_sentence.is_clause_head(token):
                pos_pattern = '[<2 /^VB.*/ | < (__ <1 cop)]'
            else:
                pos_pattern = ('<2 /^%s.*/' % token.get_gen_pos())

            return '/.*_[0-9+]/=%s %s' % (node_name, pos_pattern)
            '''

        token = sentence.tokens[node_index]
        try:
            connective_index = connective_indices.index(node_index)
            node_name = 'connective_%d' % connective_index
            node_names[node_index] = node_name
            return ('/^%s_[0-9]+$/=%s <2 /^%s.*/' % (
                        token.lemma, node_name, token.get_gen_pos()))
        except ValueError: # It's not a connective node
            try:
                steiner_index = steiner_nodes.index(node_index)
                return non_connective_pattern('steiner_%d' % steiner_index)
            except ValueError:  # It's an argument node_index
                node_name = ['cause', 'effect'][
                    token.index == effect_head.index]
                return non_connective_pattern(node_name)

    @staticmethod
    def _get_dep_edge_pattern(edge_start, edge_end, sentence):
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
    def _add_dep_edge_to_pattern(sentence, steiner_graph, pattern,
                                 node_pattern, edge_start, edge_end):
        if steiner_graph[edge_start, edge_end]: # forward edge
            edge_pattern = TRegexConnectiveModel._get_dep_edge_pattern(
                edge_start, edge_end, sentence)
            pattern = '%s < (%s %s' % (pattern, node_pattern,
                                       edge_pattern)
        else: # back edge
            edge_pattern = TRegexConnectiveModel._get_dep_edge_pattern(
                edge_end, edge_start, sentence)
            pattern = '%s %s > (%s' % (pattern, edge_pattern,
                                       node_pattern)
        return pattern

    @staticmethod
    def _get_cons_node_pattern(sentence, node_index, node_names,
                               connective_nodes, steiner_nodes, cause_node,
                               effect_node):
        tree = sentence.constituency_tree # for brevity
        node = subtree_at_index(tree, node_index)

        try:
            connective_index = connective_nodes.index(node)
            assert (isinstance(node[0][0], str)
                    or isinstance(node[0][0], unicode))
            node_name = 'connective_%d' % connective_index
            node_names[node_index] = node_name
            gen_pos = Token.POS_GENERAL.get(node.label(), node.label())
            return '(/^%s.*/=%s < %s)' % (gen_pos, node_name, node[0])
        except ValueError: # It's not a connective node
            try:
                steiner_index = steiner_nodes.index(node_index)
                node_name = 'steiner_%d' % steiner_index
                pattern = '__=%s' % node_name
            except ValueError: # It's an argument node_index
                node_name = ['cause', 'effect'][node is effect_node]
                pattern = '%s=%s' % (node.label(), node_name)
            node_names[node_index] = node_name
            return pattern

    @staticmethod
    def _add_cons_edge_to_pattern(sentence, steiner_graph, pattern,
                                  node_pattern, edge_start, edge_end):
        # TODO: Make this use <+(VP) for VPs.
        if steiner_graph[edge_start, edge_end]: # forward edge
            pattern = '%s < (%s' % (pattern, node_pattern)
        else: # back edge
            pattern = '%s > (%s' % (pattern, node_pattern)
        return pattern

    @staticmethod
    def _generate_pattern_from_steiners(sentence, steiner_graph, steiner_nodes,
                                        connective_nodes, cause, effect,
                                        path_seed_index):
        '''
        Both dependency-based and constituency-based pattern generation share
        the same algorithmic structure once the Steiner graph has been found.
        The only difference is how patterns are generated for each node/edge.

        If we're in dependency mode:
         - `cause` and `effect` are the Tokens representing the argument heads.
         - `connective_nodes` is a list of token indices.
        If we're in constituency mode:
         - `cause` and `effect` are constituency nodes spanning the argument
           annotations.
         - `connective_nodes` is a list of constituency tree nodes.
        '''

        if len(steiner_nodes) > FLAGS.tregex_max_steiners:
            logging.debug(
                "Ignoring very long pattern (sentence: %s)"
                % sentence.original_text)
            return (None, None)

        pattern = ''
        if FLAGS.tregex_pattern_type == 'dependency':
            node_pattern_fn = TRegexConnectiveModel._get_dep_node_pattern
            add_edge_fn = TRegexConnectiveModel._add_dep_edge_to_pattern
        else:
            node_pattern_fn = TRegexConnectiveModel._get_cons_node_pattern
            add_edge_fn = TRegexConnectiveModel._add_cons_edge_to_pattern

        # To generate the pattern, we start by generating one long string that
        # can be checked easily by TRegex. That'll be the biggest chunk of the
        # pattern. It consists of the longest path through the Steiner tree
        # edges.

        # Start the longest path search from a node we know is actually in the
        # tree we're looking for.
        longest_path = longest_path_in_tree(steiner_graph, path_seed_index)
        node_names = {}
        edges = [(None, longest_path[0])] + list(pairwise(longest_path))
        for edge_start, edge_end in edges:
            end_node_pattern = node_pattern_fn(
                sentence, edge_end, node_names, connective_nodes,
                steiner_nodes, cause, effect)
            if edge_start is not None:
                pattern = add_edge_fn(sentence, steiner_graph, pattern,
                                      end_node_pattern, edge_start, edge_end)
            else: # start of path
                pattern = '(%s' % end_node_pattern
        pattern += ')' * len(edges)

        # Next, we need to make sure all the edges that weren't included in the
        # longest path get incorporated into the pattern. For this, it's OK to
        # have a few colon-separated pattern segments.
        def get_named_node_pattern(node):
            try:
                return '=' + node_names[node]
            except KeyError: # Node hasn't been named and given a pattern yet
                return '(%s)' % node_pattern_fn(
                    sentence, node, node_names, connective_nodes,
                    steiner_nodes, cause, effect)

        for edge_start, edge_end in zip(*steiner_graph.nonzero()):
            if ((edge_start, edge_end) in edges
                or (edge_end, edge_start) in edges):
                continue
            start_node_pattern = get_named_node_pattern(edge_start)
            end_node_pattern = get_named_node_pattern(edge_end)
            # Link end to start using add_edge_fn, as though start were the
            # entire pattern so far. It will, in fact, be the entire pattern so
            # far after the colon.
            edge_pattern = add_edge_fn(
                sentence, steiner_graph, start_node_pattern, end_node_pattern,
                edge_start, edge_end)
            # The final paren is because the edge pattern functions don't close
            # their parens.
            pattern = '%s : (%s))' % (pattern, edge_pattern)

        node_names_to_print = [name for name in node_names.values() if name
                               not in ['cause', 'effect']]

        return pattern, node_names_to_print

    @staticmethod
    def _get_dependency_pattern(sentence, connective_tokens, cause_tokens,
                                effect_tokens):
        connective_indices = [token.index for token in connective_tokens]
        cause_head = sentence.get_head(cause_tokens)
        effect_head = sentence.get_head(effect_tokens)
        required_token_indices = list(set( # Eliminate potential duplicates
            [cause_head.index, effect_head.index] + connective_indices))

        # Once the sentence has been preprocessed, it is possible some nodes
        # will have been deleted. We make sure to delete these from the list
        # of required nodes. (We check whether each has an incoming or outgoing
        # edge.
        # TODO: remember what nodes have been deleted, so that they can be
        #       re-added as part of the connective_tokens span if the pattern
        #       matches.
        required_token_indices_to_keep = []
        for required_index in required_token_indices:
            if (sentence.edge_graph[:, required_index].nnz != 0
                or sentence.edge_graph[required_index, :].nnz != 0):
                required_token_indices_to_keep.append(required_index)
            else:
                logging.debug("Eliminating token %s from pattern"
                              % sentence.tokens[required_index])
        required_token_indices = required_token_indices_to_keep

        steiner_nodes, steiner_graph = steiner_tree(
            sentence.edge_graph, required_token_indices,
            sentence.path_costs, sentence.path_predecessors)

        path_seed_index = connective_indices[0]
        return TRegexConnectiveModel._generate_pattern_from_steiners(
            sentence, steiner_graph, steiner_nodes, connective_indices,
            cause_head, effect_head, path_seed_index)

    @staticmethod
    def _get_constituency_pattern(sentence, connective_tokens, cause_tokens,
                                  effect_tokens):
        # TODO: optimize shortest-path calculations on graph to be done only
        # once? (currently happen repeatedly in steiner_tree)
        constituency_tree = sentence.constituency_tree # for brevity
        cause_node = sentence.get_constituency_node_for_tokens(cause_tokens)
        effect_node = sentence.get_constituency_node_for_tokens(effect_tokens)
        connective_treepositions = [
            # Index includes ROOT token, so subtract 1.
            constituency_tree.leaf_treeposition(t.index - 1)
            for t in connective_tokens]
        # Leaf treepositions get us to the words themselves. We want the nodes
        # just above the words, so we lop off the ends of the positions.
        connective_nodes = [constituency_tree[position[:-1]]
                            for position in connective_treepositions]
        # Use IDs of terminal nodes so we can do quick checks for identity,
        # rather than expensive recursive equality checks.
        terminal_ids = [id(terminal) for terminal
                        in [cause_node, effect_node] + connective_nodes]
        terminal_indices = [
            i for (i, subtree) in enumerate(constituency_tree.subtrees())
            if id(subtree) in terminal_ids]
        steiner_nodes, steiner_graph = steiner_tree(
            sentence.constituency_graph, terminal_indices, directed=False)

        path_seed_index = index_of_subtree(connective_nodes[0])
        return TRegexConnectiveModel._generate_pattern_from_steiners(
            sentence, steiner_graph, steiner_nodes, connective_nodes,
            cause_node, effect_node, path_seed_index)

    @staticmethod
    def _get_pattern(sentence, connective_tokens, cause_tokens, effect_tokens):
        if FLAGS.tregex_pattern_type == 'dependency':
            return TRegexConnectiveModel._get_dependency_pattern(
                sentence, connective_tokens, cause_tokens, effect_tokens)
        else:
            return TRegexConnectiveModel._get_constituency_pattern(
                sentence, connective_tokens, cause_tokens, effect_tokens)

    def _extract_patterns(self, sentences):
        # TODO: Extend this to work with cases of missing arguments.
        self.tregex_patterns = []
        patterns_seen = set()

        preprocessed_ptb_strings, true_causation_pairs_by_sentence = (
            self._preprocess_sentences(sentences))

        if FLAGS.tregex_print_patterns:
            print 'Patterns:'
        for sentence, ptb_string in zip(sentences, preprocessed_ptb_strings):
            if FLAGS.tregex_pattern_type == 'dependency':
                sentence = sentence.substitute_dep_ptb_graph(ptb_string)
            for instance in sentence.causation_instances:
                if instance.cause != None and instance.effect is not None:
                    pattern, node_names = self._get_pattern(
                        sentence, instance.connective, instance.cause,
                        instance.effect)

                    if pattern is None:
                        continue

                    if pattern not in patterns_seen:
                        if FLAGS.tregex_print_patterns:
                            print pattern.encode('utf-8')
                            print 'Sentence:', (sentence.original_text.encode(
                                                    'utf-8'))
                            print
                        patterns_seen.add(pattern)
                        connective_lemmas = [t.lemma for t
                                             in instance.connective]
                        self.tregex_patterns.append((pattern, node_names,
                                                     connective_lemmas))

        return preprocessed_ptb_strings, true_causation_pairs_by_sentence

    #####################################
    # Running TRegex
    #####################################

    class TregexProcessorThread(threading.Thread):
        def __init__(self, sentences, ptb_strings, queue,
                     true_causation_pairs_by_sentence, *args, **kwargs):
            super(TRegexConnectiveModel.TregexProcessorThread, self).__init__(
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

                    with tempfile.NamedTemporaryFile(
                        'w', prefix='trees') as tree_file:
                        # logging.debug("Trees written to %s (pattern: %s)"
                        #              % (tree_file.name, pattern))
                        tree_file.writelines(possible_trees)
                        # Make sure the file is synced for threads to access
                        tree_file.flush()
                        self._process_pattern(
                            pattern, possible_sentences, tree_file.name,
                            possible_true_causation_pairs)
                        self.queue.task_done()
            except Queue.Empty: # no more items in queue
                return

        FIXED_TREGEX_ARGS = '-o -l -N -h cause -h effect'.split()
        def _process_pattern(self, pattern, possible_sentences, tree_file_path,
                             possible_true_causation_pairs):
            # Create output file
            with tempfile.NamedTemporaryFile(
                'w+b', prefix='matches') as self.output_file:
                # logging.debug("Processing %s to %s"
                #              % (pattern, self.output_file.name))
                # TODO: Make this use the node labels to also retrieve the
                # possible connective tokens.
                if FLAGS.tregex_pattern_type == 'dependency':
                    to_print_arg = '-u'
                else:
                    to_print_arg = '-x'

                tregex_command = (
                    [os.path.join(FLAGS.tregex_dir, 'tregex.sh'), to_print_arg]
                    + self.FIXED_TREGEX_ARGS + [pattern, tree_file_path])
                subprocess.call(tregex_command, stdout=self.output_file,
                                stderr=self.dev_null)
                self.output_file.seek(0)

                for sentence, true_causation_pairs in zip(
                    possible_sentences, possible_true_causation_pairs):
                    self._process_tregex_for_sentence(pattern, sentence,
                                                      true_causation_pairs)

                # Tell the progress reporter how far we've gotten, so that it
                # will know progress for patterns that have already finished.
                self.total_bytes_output += self.output_file.tell()

            self.output_file = None

        def _process_tregex_for_sentence(self, pattern, sentence,
                                         true_causation_pairs):
            # Read TRegex output for the sentence.
            # For each sentence, we leave the file positioned at the next
            # tree number line.
            self.output_file.readline() # skip tree num line
            next_line = self.output_file.readline().strip()
            lines = []
            while next_line:
                lines.append(next_line)
                next_line = self.output_file.readline().strip()

            # Parse TRegex output. Argument identifiers will be printed on
            # alternating lines (cause, then effect).
            for line_pair in zip(lines[0::2], lines[1::2]):
                if FLAGS.tregex_pattern_type == 'dependency':
                    index_pair = [int(line.split("_")[-1])
                                  for line in line_pair]
                    index_pair = tuple(sorted(index_pair))
                    token_1, token_2 = [sentence.tokens[i] for i in index_pair]
                else: # constituency
                    # We need to use treepositions, not subtrees, because this
                    # is how TRegex gives match positions.
                    all_treepositions = (sentence.constituency_tree
                                          .treepositions())
                    arg_treeposition_indices = [int(line.split(":")[1])
                                                for line in line_pair]
                    # TRegex node positions start from 1, so we need to
                    # subtract 1 to get proper treeposition indices.
                    arg_constituent_nodes = [
                        sentence.constituency_tree[all_treepositions[pos - 1]]
                        for pos in arg_treeposition_indices]
                    arg_constituent_heads = [sentence.constituent_heads[node]
                                             for node in arg_constituent_nodes]
                    token_1, token_2 = TRegexConnectiveStage.normalize_order(
                        [sentence.get_token_for_constituency_node(head)
                         for head in arg_constituent_heads])
                    index_pair = tuple(t.index for t in [token_1, token_2])

                # Mark sentence if possible connective is present.
                in_gold = index_pair in true_causation_pairs
                possible = PossibleCausation(
                    token_1, token_2, pattern, in_gold)
                # THIS IS THE ONLY LINE THAT MUTATES SHARED DATA.
                # It is thread-safe, because lists are thread-safe, and
                # we never reassign sentence.possible_causations.
                sentence.possible_causations.append(possible)

        def get_progress(self):
            try:
                return self.total_bytes_output + self.output_file.tell()
            except (AttributeError, IOError, ValueError):
                # AttributeError indicates that self.output_file was None.
                # IOError/ValueError indicate that we managed to ask for file
                # size just after the file was closed. Either way, that means
                # that now the total number of bytes has been recorded.
                return self.total_bytes_output

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


class TRegexConnectiveStage(PairwiseCausalityStage):
    def __init__(self, name):
        super(TRegexConnectiveStage, self).__init__(
            print_test_instances=FLAGS.tregex_print_test_instances, name=name,
            models=[TRegexConnectiveModel(part_type=ParsedSentence)])

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
            assert (None, None) not in expected_pairs
            tp, fp, fn = self.match_causation_pairs(
                expected_pairs, predicted_pairs, self.tp_pairs, self.fp_pairs,
                self.fn_pairs, self.all_instances_metrics)

            self.pairwise_only_metrics.tp += tp
            self.pairwise_only_metrics.fp += fp
            fns_to_ignore = sum(1 for pair in expected_pairs if None in pair)
            self.pairwise_only_metrics.fn += fn - fns_to_ignore
