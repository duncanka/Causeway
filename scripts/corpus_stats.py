from __future__ import print_function
from collections import Counter, defaultdict
from itertools import chain
from math import log
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter
import numpy as np
import re
from warnings import warn

from causeway.because_data import (CausalityStandoffReader, CausationInstance,
                                   OverlappingRelationInstance)
from causeway.because_data.iaa import stringify_connective
from nlpypline.data import DependencyPath, Token
from nlpypline.data.io import DirectoryReader
from nlpypline.util import listify, partition

from read_all import read_all # just for easy access

def utopia_context():
    return mpl.rc_context(rc={'font.family': 'serif',
                              'font.serif': 'Utopia',
                              'text.latex.preamble': '\usepackage{fourier}',
                              'text.usetex': True})

def not_contiguous(instance):
    connective = instance.connective
    if len(connective) == 2 and ((connective[0].pos in Token.VERB_TAGS
                                  and connective[1].pos in ['IN', 'TO'])):
                                 # or connective[0].lemma == 'be'
                                 # or connective[1].lemma == 'be'):
        return False

    start = connective[0].index
    for conn_token in connective[1:]:
        if conn_token.index != start + 1:
            # print(instance)
            return True
        else:
            start = conn_token.index
    return False


def mwe(instance):
    connective = instance.connective
    if len(connective) == 2 and ((connective[0].pos in Token.VERB_TAGS
                                  and connective[1].pos in ['IN', 'TO'])):
                                 # or connective[0].lemma == 'be'
                                 # or connective[1].lemma == 'be'):
        return False

    if len(connective) > 1:
        # print(instance)
        return True
    return False


def count(documents, criterion, print_matching=False):
    matching = 0
    for d in documents:
        for s in d.sentences:
            for i in s.causation_instances:
                if criterion(i):
                    matching += 1
                    if print_matching:
                        print(i)
    return matching


def count_connectives(documents):
    counts = Counter()
    for d in documents:
        for s in d.sentences:
            counts += Counter(' '.join([t.lemma for t in i.connective])
                              for i in s.causation_instances)
    return counts


def count_with_overlaps(documents, exclude=[]):
    counts = Counter()
    for d in documents:
        for s in d.sentences:
            connectives_to_categories = defaultdict(set)
            for i in s.causation_instances:
                conn = tuple(i.connective)
                if exclude and set(t.lemma for t in conn) in exclude:
                    continue
                connectives_to_categories[conn] = set(
                    [i.get_interpretable_type()])
            for i in s.overlapping_rel_instances:
                conn = tuple(i.connective)
                if exclude and set(t.lemma for t in conn) in exclude:
                    continue
                connectives_to_categories[conn].update(
                    i.get_interpretable_type())
            counts += Counter(tuple(v) for v in
                              connectives_to_categories.values())
    return counts


def count_conns_with_overlaps(documents):
    counts = defaultdict(Counter)
    for d in documents:
        for s in d.sentences:
            connectives_to_categories = defaultdict(set)
            for i in s.causation_instances:
                connectives_to_categories[tuple(i.connective)] = set(
                    [i.get_interpretable_type()])
            for i in s.overlapping_rel_instances:
                connectives_to_categories[tuple(i.connective)].update(
                    i.get_interpretable_type())
            for conn, categories in connectives_to_categories.iteritems():
                counts[tuple(categories)][' '.join(t.lemma for t in conn)] += 1
    return counts


def count_from_files(paths, criterion, print_matching=False, recursive=False):
    reader = DirectoryReader((CausalityStandoffReader.FILE_PATTERN,),
                             CausalityStandoffReader(), recursive)
    paths = listify(paths)
    total = 0
    for path in paths:
        reader.open(path)
        docs = reader.get_all()
        total += count(docs, criterion, print_matching)
    return total


def arg_deps(instances, pairwise=True):
    cause_deps = Counter()
    effect_deps = Counter()
    for i in instances:
        if pairwise and not (i.cause and i.effect):
            continue

        sentence = i.sentence
        cause, effect = i.cause, i.effect
        for arg, deps in zip([cause, effect], [cause_deps, effect_deps]):
            if arg:
                arg_head = sentence.get_head(arg)
                if arg_head:
                    incoming_dep, parent = sentence.get_most_direct_parent(arg_head)
                    deps[incoming_dep] += 1
                # Else skip invalid head
    return cause_deps, effect_deps


def arg_lengths(instances, pairwise=True):
    cause_lengths = Counter()
    effect_lengths = Counter()
    for i in instances:
        if pairwise and not (i.cause and i.effect):
            continue

        sentence = i.sentence
        cause, effect = i.cause, i.effect
        for arg, sizes in zip([cause, effect], [cause_lengths, effect_lengths]):
            sizes[len(arg)] += 1
    return cause_lengths, effect_lengths


def plot_arg_lengths(cause_lengths, effect_lengths):
    min_bin, max_bin = 1, 22
    bins = range(min_bin, max_bin)
    bins[-1] -= 0.0001
    causes, effects = [list(chain.from_iterable([i] * l[i] for i in bins))
                       for l in cause_lengths, effect_lengths]
    cause_y, bin_edges = np.histogram(causes, bins=bins)
    effect_y, _ = np.histogram(effects, bins=bins)

    plt.plot(bin_edges[:-1], cause_y, '--', color='#3c6090', marker='o')
    plt.plot(bin_edges[:-1], effect_y, '-', color='#e84330', marker='s')

    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    with utopia_context():
        plt.tick_params(axis='both', labelsize=11)
        plt.xlim(0, max_bin - 1.5)
        # 0 is an uninteresting tick, but 1 is relevant.
        ax.xaxis.set_major_locator(FixedLocator([1, 5, 10, 15, 20]))
        plt.xlabel('Argument length (in tokens)', fontsize=13, labelpad=12)
        plt.ylabel('Count', fontsize=13, labelpad=12)
        plt.text(2.8, 190, 'Causes', color='#3c6090', fontsize=15)
        plt.text(7.3, 120, 'Effects', color='#e84330', fontsize=15)

        plt.tight_layout()

        fig = plt.gcf()
        size = fig.get_size_inches()
        fig.set_size_inches(size[0], size[1] * 0.8)

        #plt.show(False)
        plt.savefig('/home/jesse/Documents/Work/Research/My Publications/Thesis/tagging/arg_lengths.pdf')


def length_median(arg_lengths):
    return np.median(list(chain.from_iterable([i] * arg_lengths[i]
                                              for i in arg_lengths)))


def latex_for_corpus_counts(counts):
    causation_types = CausationInstance.CausationTypes[:-1] # skip Inference
    def latex_type_count(count):
        if count == 0:
            return r'\textcolor{darkgray}{-}'
        else:
            return str(count)

    del counts[('UNKNOWN',)]

    # First, the non-overlapping instances.
    print('None', end='')
    total = 0
    for cause_type in causation_types:
        type_count = sum([v for k, v in counts.items() if k == (cause_type,)])
        print(' &', type_count, end='')
        total += type_count
    # All causal
    print(' &', latex_type_count(total), end='')
    # No non-causals with no overlapping relation
    print(' &', latex_type_count(0), end='')
    print(' &', total, end=' \\\\\n')

    for overlapping_type in OverlappingRelationInstance.RelationTypes:
        total = 0
        print(r'\formal{%s}' % overlapping_type.replace('_', '/'), end = '')
        for cause_type in causation_types:
            type_count = sum(
                [v for k, v in counts.items()
                 if overlapping_type in k and cause_type in k])
            print(' &', latex_type_count(type_count), end='')
            total += type_count
        # Total across all causals
        print(' &', latex_type_count(total), end='')
        # Non-causal
        type_count = sum([v for k, v in counts.items()
                          if not any(t in k for t in causation_types)
                          and overlapping_type in k])
        print(' &', latex_type_count(type_count), end='')
        total += type_count
        print(' &', latex_type_count(total), end=' \\\\\n')

    print(r'\midrule')
    print(r'\textbf{Total}', end='')
    total = 0
    for cause_type in causation_types:
        type_count = sum([v for k, v in counts.items() if cause_type in k])
        print(' &', latex_type_count(type_count), end='')
        total += type_count
    print(' &', latex_type_count(total), end='')
    non_causal_count = sum([v for k, v in counts.items()
                            if not any(t in k for t in causation_types)])
    print(' &', latex_type_count(non_causal_count), end='')
    total += non_causal_count
    print(' &', latex_type_count(total), end=r' \\')


def pattern_saturation(documents, num_folds=20, num_increments=20):
    sentences = list(chain.from_iterable(d.sentences for d in documents))
    xs = np.linspace(0, 1, num_increments + 1)
    ys = np.empty((num_folds, num_increments + 1))
    ys[:, 0] = 0 # At 0% of sentences, we always have 0% of the patterns.

    for fold in range(num_folds):
        np.random.shuffle(sentences)
        patterns_seen = set()
        increments = partition(sentences, num_increments)
        for i, increment in enumerate(increments):
            for sentence in increment:
                for causation in sentence.causation_instances:
                    patterns_seen.add(stringify_connective(causation))
            ys[fold, i + 1] = len(patterns_seen)
    averages = np.average(ys, 0)

    tmp_A = []
    for i in range(1, len(xs)):
        tmp_A.append([np.log(xs[i])**2, np.log(xs[i]), 1])
    b = np.matrix(averages[1:]).T
    A = np.matrix(tmp_A)
    fit = (A.T * A).I * A.T * b
    print(fit)
    # errors = b - A * fit
    # residual = np.linalg.norm(errors)

    fit_x = np.linspace(0, 4, 2000)
    fit_y = [float(fit[0]) * np.log(x)**2 + float(fit[1]) * np.log(x)
             + float(fit[2]) for x in fit_x]

    with utopia_context():
        plt.tick_params(axis='both', labelsize=11)

        plt.fill_between([1, 4], 0, 380, color='gray', alpha=0.1, lw=0)
        plt.plot(xs, averages, color='black')
        plt.plot(fit_x, fit_y, color='orange', dashes=[3,4], alpha=0.8)
        plt.xlabel(r'\% of sentences in corpus', fontsize=13, labelpad=12)
        plt.ylabel(r'\# of patterns', fontsize=13, labelpad=12)

        fig = plt.gcf()
        size = fig.get_size_inches()
        fig.set_size_inches(size[0]*1.25, size[1])

        ax = plt.gca()
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: '{:.0%}'.format(x)))

        plt.tight_layout()
        # plt.show(False)
        plt.savefig('/home/jesse/Documents/Work/Research/My Publications/Thesis/repr_annot/saturation.pdf')


def count_connectives_remapped(docs):
    to_remap = {'for too to': 'too for to', 'for too': 'too for',
                'that now': 'now that', 'to for': 'for to', 'give': 'given',
                'citizen-sparked': 'spark', 'encouraging': 'encourage',
                'have to for to': 'for to have to', 'thank to': 'thanks to',
                'on grounds of': 'on ground of', 'precipitating': 'precipitate',
                'to need': 'need to', 'to need to': 'need to to',
                'to take': 'take to', 'HELPS': 'help', 'helps': 'help',
                'on grounds that': 'on ground that'
    }
    instances = chain.from_iterable(chain.from_iterable(
        [s.causation_instances for s in doc.sentences] for doc in docs))
    stringified = [stringify_connective(causation).lower()
                   for causation in instances]
    for s, inst in zip(stringified, instances):
        assert s != 'without '
    return Counter([to_remap.get(s, s) for s in stringified])


V2_REMAPPINGS = {
    'for too to': 'too for to', 'for too': 'too for', 'give': 'given',
    'citizen-sparked': 'spark', 'have to for to': 'for to have to',
    'thank to': 'thanks to', 'on grounds of': 'on ground of',
    'HELPS': 'help', 'to take': 'take to', 'Therefore': 'therefore',
    'precipitating': 'precipitate', 'on grounds': 'on ground',
    'TO': 'to', 'encouraging': 'encourage', 'to need': 'need to',
    'to need to': 'need to to'}

def entropy_by_pattern(docs):
    connective_counts = count_connectives(docs)
    potential_connectives = Counter()
    connective_regexes = {conn: re.compile('.*'.join([r'\b{}\b'.format(word)
                                                      for word in conn.split()]))
                          for conn in connective_counts}
    for doc in docs:
        for sentence in doc:
            for conn, regex in connective_regexes.iteritems():
                sent_str = ' '.join([t.lemma for t in sentence.tokens[1:]])
                hits = regex.findall(sent_str)
                potential_connectives[conn] += len(hits)

    for merge_from, merge_into in V2_REMAPPINGS.iteritems():
        try:
            connective_counts[merge_into] = connective_counts.pop(merge_from)
            potential_connectives[merge_into] = potential_connectives.pop(
                merge_from)
        except KeyError:
            warn("No such connective: %s" % merge_from)

    probabilities = {}
    for conn in connective_counts:
        potentials = potential_connectives[conn]
        true_count = connective_counts[conn]
        if true_count > potentials: # Happens when instances crossed sentences
            potentials = true_count
            warn("Odd match count for %s" % conn)
        probabilities[conn] = true_count / float(potentials)

    # Entropy formula
    entropies = {conn: 0.0 if prob in [0.0, 1.0]
                       else -sum(p * log(p, 2) for p in [prob, 1 - prob])
                 for conn, prob in probabilities.iteritems()}
    # Give and given are indistinguishable.
    entropies['give'] = entropies['given']
    return entropies


# Based on https://stackoverflow.com/a/42295369
def entropy(counter):
    total_counted = float(sum(counter.values()))
    prob_dict = {k: v / total_counted for k, v in counter.iteritems()}
    probs = np.array(list(prob_dict.values()))
    return - probs.dot(np.log2(probs))

def arg_path_diversity_by_pattern(instances, outgoing_only=False):
    cause_paths = defaultdict(Counter)
    effect_paths = defaultdict(Counter)

    cause_path_lens = defaultdict(list)
    effect_path_lens = defaultdict(list)

    for i in instances:
        sentence = i.sentence
        cause, effect = i.cause, i.effect
        connective_str = stringify_connective(i).lower()
        connective_str = V2_REMAPPINGS.get(connective_str, connective_str)
        for arg, paths, path_lens in zip([cause, effect],
                                         [cause_paths, effect_paths],
                                         [cause_path_lens, effect_path_lens]):
            if arg:
                arg_head = sentence.get_head(arg)
                if arg_head:
                    conn_head = sentence.get_closest_of_tokens(arg_head, i.connective)[0]
                    path = sentence.extract_dependency_path(conn_head, arg_head)
                    path_lens[connective_str].append(len(path))
                    if outgoing_only:
                        path = DependencyPath(path.start, [path[0]])
                    paths[connective_str][str(path)] += 1
                # Else skip invalid head

    cause_effect_path_entropies = [{k: entropy(v) for k, v in paths.iteritems()}
                                   for paths in [cause_paths, effect_paths]]
    cause_effect_path_len_stds = [{k: np.std(v) for k, v in lens.iteritems()}
                                  for lens in [cause_path_lens, effect_path_lens]]
    return cause_effect_path_entropies + cause_effect_path_len_stds
