def not_contiguous(instance):
    connective = instance.connective
    if len(connective) == 2 and ((connective[0].pos in Token.VERB_TAGS and connective[1].pos in ['IN', 'TO']) or connective[0].lemma == 'be' or connective[1].lemma == 'be'):
        return False

    start = connective[0].index
    for conn_token in connective[1:]:
        if conn_token.index != start + 1:
            print instance
            return True
        else:
            start = conn_token.index


def mwe(instance):
    connective = instance.connective
    if len(connective) == 2 and ((connective[0].pos in Token.VERB_TAGS and connective[1].pos in ['IN', 'TO']) or connective[0].lemma == 'be' or connective[1].lemma == 'be'):
        return False

    if len(connective) > 1:
        print instance
        return True
    return False
