#!/usr/bin/env python

from glob import glob
import random
import os
import sys
import subprocess
import shutil

from util import partition

src, target, pct = sys.argv[1:]
pct = float(pct)


src_files = glob(os.path.join(src, '*.txt'))
random.shuffle(src_files)

sentence_counts = []
total_sentences = 0
for src_file in src_files:
    parse_file_name = src_file[:-4] + '.parse'
    proc = subprocess.Popen(["/bin/bash", "-c", "echo $(( $(grep '^$' %s | wc -l) / 4))" % parse_file_name],
                            stdout=subprocess.PIPE)
    sentence_count = int(proc.communicate()[0])
    print parse_file_name, ':', sentence_count
    sentence_counts.append(sentence_count)
    total_sentences += sentence_count
sentence_threshold = int(pct * total_sentences)

to_keep = []
sentences_kept = 0
for src_file, sentence_count in zip(src_files, sentence_counts):
    if sentences_kept > sentence_threshold:
        print sentences_kept, 'sentences kept; threshold was', sentence_threshold
        break
    to_keep.append(src_file)
    sentences_kept += sentence_count

try:
    os.makedirs(target)
except OSError: # already exists
    pass
for src_file in to_keep:
    shutil.copy2(src_file, target)
    shutil.copy2(src_file[:-4] + '.parse', target)
    shutil.copy2(src_file[:-4] + '.ann', target)
