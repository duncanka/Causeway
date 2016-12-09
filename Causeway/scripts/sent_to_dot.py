#!/usr/bin/env python
import sys
from tempfile import NamedTemporaryFile

from parse import run_parser
from parse_to_dot import make_dot

if __name__ == '__main__':
    if len(sys.argv) > 1:
        parser_path = sys.argv[1]
    else:
        parser_path = '../../../stanford-parser/'

    # Read from stdin
    print "Enter your sentence to parse below:"
    sentence = sys.stdin.readline()
    parse = run_parser(sentence, parser_path)
    print parse

    with NamedTemporaryFile(mode='r+', prefix='parse') as f:
        f.write(parse)
        f.file.seek(0)
        make_dot(f.file, f.name)
