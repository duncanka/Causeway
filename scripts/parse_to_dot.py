#!/usr/bin/env python

import re
import subprocess
import os
import sys

line_pattern = re.compile(r'([^\(]*)\((.*)\, (.*)\)')

def make_dot(parse_file, filename):
    dot_str = 'digraph G {\n'
    for line in parse_file.readlines():
        line = line.strip()
        if not line: # Blank line
            continue
        (rel, arg1, arg2) = line_pattern.match(line).group(1,2,3)
        arg1 = arg1.replace('-', '_')
        arg2 = arg2.replace('-', '_')
        dot_str += '    "%s" -> "%s"[label="%s"]\n' % (arg1, arg2, rel)
    dot_str += '}\n'
    print dot_str

    fname_base = os.path.splitext(filename)[0]
    img_fname = "%s.png" % fname_base
    dot_proc = subprocess.Popen(["dot", "-Tpng", "-o%s" % img_fname],
                                stdin=subprocess.PIPE)
    (std, err) = dot_proc.communicate(input=dot_str)
    if err:
        sys.stderr.write(err)
    else:
        with open(os.devnull, 'w') as null:
            subprocess.Popen(['xdg-open', img_fname], stderr=null)

if __name__ == '__main__':
    if len(sys.argv) > 1:
        fname = sys.argv[1]
    else:
        fname = '/tmp/parse-stdin'

    # Read from stdin
    print "Enter your dependency parse document below:"
    make_dot(sys.stdin, fname)
