from data import Token
from data.io import CausalityStandoffReader, DirectoryReader
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


def count(documents, criterion):
    return sum([sum([len([i for i in s.causation_instances if criterion(i)])
                     for s in d.sentences])
                for d in documents])


def count_from_files(paths, criterion):
    reader = DirectoryReader((CausalityStandoffReader.FILE_PATTERN,),
                             CausalityStandoffReader())
    paths = listify(paths)
    total = 0
    for path in paths:
        reader.open(path)
        docs = reader.get_all()
        total += count(docs, criterion)
    return total
