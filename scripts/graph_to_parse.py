from StringIO import StringIO
from parse_to_dot import make_dot

def parsify(sentence):
    lines = []
    for (start, end), label in sentence.edge_labels.items():
        lines.append('%s(%s_%s, %s_%s)' % (
            label, sentence.tokens[start].lemma, start,
            sentence.tokens[end].lemma, end))
    return lines

def visualize_sentence(sentence):
    make_dot(StringIO('\n'.join(parsify(sentence))), 'tmp%d' % id(sentence))
