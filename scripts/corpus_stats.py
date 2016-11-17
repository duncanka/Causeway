from collections import Counter
from itertools import chain
import matplotlib.pyplot as plt

from causeway.because_data import CausalityStandoffReader
from data import Token
from data.io import DirectoryReader
from util import listify


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
            # print instance
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
        # print instance
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
                        print i
    return matching


def count_from_files(paths, criterion, print_matching=False):
    reader = DirectoryReader((CausalityStandoffReader.FILE_PATTERN,),
                             CausalityStandoffReader())
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
    min_bin, max_bin = 1, 21
    bins = range(min_bin, max_bin)
    causes, effects = [list(chain.from_iterable([i] * l[i] for i in bins))
                       for l in cause_lengths, effect_lengths]
    plt.hist(causes, bins=bins, color='#4b79b4')
    plt.hist(effects, bins=bins, color='#FA8072', alpha=0.7)

    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    plt.tick_params(axis='both', labelsize=15, length=0)
    plt.xlim(min_bin, max_bin-1)
    plt.xlabel('Argument length', fontsize=16)
    plt.ylabel('Count', fontsize=16)
    plt.text(3.3, 125, 'Causes', color='#4b79b4', fontsize=18)
    plt.text(7.3, 85, 'Effects', color='#f96353', fontsize=18)
    plt.show(False)
