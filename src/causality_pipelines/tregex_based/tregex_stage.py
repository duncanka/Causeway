from gflags import DEFINE_string, DEFINE_bool, FLAGS, DuplicateFlagError, DEFINE_integer, DEFINE_enum
import threading
import logging
from math import log10
from os import path
import Queue
import subprocess
import tempfile
import time

from data import ParsedSentence, Token
from pipeline import Stage
from pipeline.models import Model
from causality_pipelines import PossibleCausation, IAAEvaluator
from util import pairwise, igroup
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

class TRegexConnectiveModel(Model):
    def __init__(self, *args, **kwargs):
        super(TRegexConnectiveModel, self).__init__(*args, **kwargs)
        self.tregex_patterns = []
        # Internal hackery properties, used for training.
        self._ptb_strings = None

    def train(self, sentences):
        ptb_strings = self._extract_patterns(sentences)
        # Dirty hack to avoid redoing all the preprocessing when test() is
        # called to provide input to the next stage.
        self._ptb_strings = ptb_strings

    def test(self, sentences):
        logging.info('Tagging possible connectives...')
        start_time = time.time()

        if self._ptb_strings is not None:
            ptb_strings = self._ptb_strings
            self._ptb_strings = None
        else:
            ptb_strings = self._preprocess_sentences(sentences)

        # Interacting with the TRegex processes is heavily I/O-bound, plus we
        # really want multiple TRegex processes running in parallel, so we farm
        # out patterns to worker threads.

        # Queue up the patterns
        total_estimated_bytes = 0
        queue = Queue.Queue()
        for (pattern, connective_labels, connective_lemmas
             ) in self.tregex_patterns:
            possible_sentence_indices = self._filter_sentences_for_pattern(
                sentences, pattern, connective_lemmas)
            queue.put_nowait((pattern, connective_labels,
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
                sentences, ptb_strings, queue)
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
        for sentence in sentences:
            sentence.possible_causations = []
            if FLAGS.tregex_pattern_type == 'dependency':
                ptb_strings.append(sentence.dep_to_ptb_tree_string() + '\n')
            else:
                ptb_strings.append(sentence.constituency_tree.pformat() + '\n')

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
                path.join('causality_pipelines', 'tregex_based', 'tsurgeon_dep',
                          script_name) + '.ts'
                for script_name in tsurgeon_script_names]

            with tempfile.NamedTemporaryFile('w', delete=False) as tree_file:
                encoded_strings = [s.encode('utf-8') for s in ptb_strings]
                tree_file.writelines(encoded_strings)
                tree_file.flush()
                with tempfile.NamedTemporaryFile(
                    'w+b', delete=False) as surgeried_file:
                    tsurgeon_command = (
                        ([path.join(FLAGS.tregex_dir, 'tsurgeon.sh'), '-s',
                          '-treeFile', tree_file.name]
                         + tsurgeon_script_names))
                    subprocess.call(tsurgeon_command, stdout=surgeried_file,
                                    stderr=(TRegexConnectiveModel
                                            .TregexProcessorThread.dev_null))
                    surgeried_file.seek(0)
                    ptb_strings = surgeried_file.readlines()
        else:
            # Temporary measure until we get TSurgeon scripts updated for
            # constituency parses: don't do any real preprocessing.
            # TODO: Implement constituency scripts, and move TSurgeon-running
            # code to be shared.
            pass

        logging.info('Done preprocessing.')
        return ptb_strings

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
        longest_path = list(
            longest_path_in_tree(steiner_graph, path_seed_index))
        if FLAGS.tregex_pattern_type == 'dependency':
            # Normalize the path so that we don't end up thinking the reverse
            # path is a totally different pattern: always put the cause first.
            try:
                if (longest_path.index(cause.index) >
                    longest_path.index(effect.index)):
                    longest_path = longest_path[::-1]
            except ValueError:
                # TODO: Should we normalize in some other way if both args
                # aren't in the longest path?
                pass
        # TODO: implement this for constituency?

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

        # All connective node IDs should be printed by TRegex.
        node_names_to_print = [name for name in node_names.values()
                               if name.startswith('connective')]

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

        preprocessed_ptb_strings = self._preprocess_sentences(sentences)

        logging.info('Extracting patterns...')
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
        logging.info('Done extracting patterns.')

        return preprocessed_ptb_strings

    #####################################
    # Running TRegex
    #####################################

    class TregexProcessorThread(threading.Thread):
        def __init__(self, sentences, ptb_strings, queue, *args, **kwargs):
            super(TRegexConnectiveModel.TregexProcessorThread, self).__init__(
                *args, **kwargs)
            self.sentences = sentences
            self.ptb_strings = ptb_strings
            self.queue = queue
            self.output_file = None
            self.total_bytes_output = 0

        dev_null = open('/dev/null', 'w')

        def run(self):
            try:
                while(True):
                    (pattern, connective_labels, possible_sentence_indices) = (
                        self.queue.get_nowait())
                    if not possible_sentence_indices: # no sentences to scan
                        self.queue.task_done()
                        continue

                    possible_trees = [self.ptb_strings[i]
                                      for i in possible_sentence_indices]
                    possible_sentences = [self.sentences[i]
                                          for i in possible_sentence_indices]

                    with tempfile.NamedTemporaryFile(
                        'w', prefix='trees') as tree_file:
                        # logging.debug("Trees written to %s (pattern: %s)"
                        #              % (tree_file.name, pattern))
                        tree_file.writelines(possible_trees)
                        # Make sure the file is synced for threads to access
                        tree_file.flush()
                        self._process_pattern(
                            pattern, connective_labels, possible_sentences,
                            tree_file.name)
                    self.queue.task_done()
            except Queue.Empty: # no more items in queue
                return

        FIXED_TREGEX_ARGS = '-o -l -N -h cause -h effect'.split()
        def _process_pattern(
            self, pattern, connective_labels, possible_sentences,
            tree_file_path):
            # Create output file
            with tempfile.NamedTemporaryFile(
                'w+b', prefix='matches') as self.output_file:
                # logging.debug("Processing %s to %s"
                #              % (pattern, self.output_file.name))
                if FLAGS.tregex_pattern_type == 'dependency':
                    output_type_arg = '-u'
                else:
                    output_type_arg = '-x'

                connective_printing_args = []
                for connective_label in connective_labels:
                    connective_printing_args.extend(['-h', connective_label])

                tregex_command = (
                    [path.join(FLAGS.tregex_dir, 'tregex.sh'), output_type_arg]
                    + self.FIXED_TREGEX_ARGS + connective_printing_args
                    + [pattern, tree_file_path])
                subprocess.call(tregex_command, stdout=self.output_file,
                                stderr=self.dev_null)
                self.output_file.seek(0)

                for sentence in possible_sentences:
                    self._process_tregex_for_sentence(
                        pattern, connective_labels, sentence)

                # Tell the progress reporter how far we've gotten, so that it
                # will know progress for patterns that have already finished.
                self.total_bytes_output += self.output_file.tell()

            self.output_file = None

        @staticmethod
        def _get_constituency_token_from_tregex_line(line, sentence,
                                                     all_treepositions):
            # We need to use treepositions, not subtrees, because this
            # is how TRegex gives match positions.
            treeposition_index = int(line.split(":")[1])
            node = sentence.constituency_tree[
                all_treepositions[treeposition_index - 1]]
            head = sentence.constituent_heads[node]
            return sentence.get_token_for_constituency_node(head)
        
        @staticmethod
        def _get_dependency_token_from_tregex_line(line, sentence):
            token_index = int(line.split("_")[-1])
            return sentence.tokens[token_index]

        def _process_tregex_for_sentence(self, pattern, connective_labels,
                                         sentence):
            # Read TRegex output for the sentence.
            # For each sentence, we leave the file positioned at the next
            # tree number line.
            self.output_file.readline() # skip tree num line
            next_line = self.output_file.readline().strip()
            lines = []
            while next_line:
                lines.append(next_line)
                next_line = self.output_file.readline().strip()
                
            true_connectives = {
                tuple(instance.connective): instance
                for instance in sentence.causation_instances
                if instance.cause and instance.effect # limit to pairwise
            }

            # Parse TRegex output. Argument and connective identifiers will be
            # printed in batches of 2 + k, where k is the connective length.
            # The first two printed will be cause/effect; the remainder are
            # connectives.
            batch_size = 2 + len(connective_labels)
            for match_lines in igroup(lines, batch_size):
                # TODO: If argument heads overlap with connective words, we get
                # a problem where the pattern fails to include one of them,
                # which means the match always excludes something important and
                # we end up with the wrong number of match lines. Fix this
                # (probably by generating class patterns that can store extra
                # data like argument mappings).
                if None in match_lines:
                    logging.warn("Skipping invalid TRegex match: %s", lines)
                    continue
                arg_lines = match_lines[:2]
                connective_lines = match_lines[2:]

                if FLAGS.tregex_pattern_type == 'dependency':
                    cause, effect = [
                        self._get_dependency_token_from_tregex_line(line,
                                                                    sentence)
                        for line in arg_lines]
                    connective = [
                        self._get_dependency_token_from_tregex_line(line,
                                                                    sentence)
                        for line in connective_lines]
                else: # constituency
                    all_treepositions = (sentence.constituency_tree
                                         .treepositions())
                    cause, effect = [
                        self._get_constituency_token_from_tregex_line(
                            line, sentence, all_treepositions)
                        for line in arg_lines]
                    connective = [
                        self._get_constituency_token_from_tregex_line(
                            line, sentence, all_treepositions)
                        for line in connective_lines]

                # TODO: Make this eliminate duplicate PossibleCausations on
                # the same connective words, like regex pipeline does.
                possible = PossibleCausation(
                    [pattern], connective,
                    true_connectives.get(tuple(connective), None),
                    [cause], [effect])
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


class TRegexConnectiveStage(Stage):
    def __init__(self, name):
        super(TRegexConnectiveStage, self).__init__(
            name=name,
            models=[TRegexConnectiveModel(part_type=ParsedSentence)])
        self.pairwise_only_metrics = None # used during evaluation

    PRODUCED_ATTRIBUTES = ['possible_causations']
    
    def _extract_parts(self, sentence, is_train):
        return [sentence]

    def _make_evaluator(self):
        # TODO: provide both pairwise and non-pairwise stats
        # TODO: figure out why this doesn't print test instances
        return IAAEvaluator(False, False, FLAGS.tregex_print_test_instances,
                            True, True, True)
