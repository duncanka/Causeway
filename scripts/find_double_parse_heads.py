#!/usr/bin/env python

import sys
from data.readers import StanfordParsedSentenceReader

for filepath in sys.argv[1:]:
    reader = StanfordParsedSentenceReader()
    reader.open(filepath)
    sentences = reader.get_all()
    reader.close()

    for sentence in sentences:
        for token in sentence.tokens[1:]:
            num_edges = 0
            incoming = sentence.edge_graph[:, token.index].nonzero()[0]
            for start_index in incoming:
                if sentence.edge_labels[(start_index, token.index)] != 'ref':
                    num_edges += 1
            # Only print if the node has >1 incoming edge and at least one
            # outgoing edge.
            if num_edges > 1 and sentence.edge_graph[token.index].nnz > 0:
                print sentence.source_file_path + ':', sentence.original_text
                break
