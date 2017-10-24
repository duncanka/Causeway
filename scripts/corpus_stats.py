from __future__ import print_function
from collections import Counter, defaultdict
from itertools import chain
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator
import numpy as np

from causeway.because_data import (CausalityStandoffReader, CausationInstance,
                                   OverlappingRelationInstance)
from causeway.because_data.iaa import stringify_connective
from nlpypline.data import Token
from nlpypline.data.io import DirectoryReader
from nlpypline.util import listify, partition


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
            counts += Counter(tuple(t.lemma for t in i.connective)
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
                incoming_dep, parent = sentence.get_most_direct_parent(arg_head)
                deps[incoming_dep] += 1
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
    mpl.rc('font',**{'family':'serif','serif':['Times']})
    mpl.rc('text', usetex=True)

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

    plt.tick_params(axis='both', labelsize=18, length=0)
    plt.xlim(0, max_bin - 1.5)
    # 0 is an uninteresting tick, but 1 is relevant.
    ax.xaxis.set_major_locator(FixedLocator([1, 5, 10, 15, 20]))
    plt.xlabel('Argument length (in tokens)', fontsize=20)
    plt.ylabel('Count', fontsize=20)
    plt.text(3.3, 125, 'Causes', color='#3c6090', fontsize=22)
    plt.text(7.3, 85, 'Effects', color='#e84330', fontsize=22)
    plt.tight_layout()
    plt.show(False)


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

    plt.plot(xs, averages)
    plt.xlabel('% of sentences in corpus', fontsize=18)
    plt.ylabel('# of patterns', fontsize=18)
    plt.tight_layout()
    plt.show(False)


def count_connectives_remapped(instances):
    to_remap = {'for too to': 'too for to', 'for too': 'too for',
                'that now': 'now that', 'to for': 'for to', 'give': 'given',
                'citizen-sparked': 'spark', 'encouraging': 'encourage',
                'have to for to': 'for to have to', 'thank to': 'thanks to',
                'on ground of': 'on grounds of', 'precipitating': 'precipitate',
                'to need': 'need to', 'to need to': 'need to to',
                'to take': 'take to', 'reason be': 'reason',
                'result of': 'result' # from old corpus
    }
    stringified = [stringify_connective(causation).lower()
                   for causation in instances]
    for s, inst in zip(stringified, instances):
        assert s != 'without '
    return Counter([to_remap.get(s, s) for s in stringified])
