#!/usr/bin/env python

from glob import glob
import os
import random
import shutil
import sys

DATA_DIR = '/home/jesse/Documents/Work/Research/Data/nltk_data/corpora/treebank/combined/'
OUTPUT_DIR = '/var/www/brat/data/Jeremy/PTB'
MAX_SENTENCES = 1400

all_filepaths = []
ptb_sections = [str(k) for k in range(2, 25)] # Omit 0 and 1 (supposedly worse)
for section in ptb_sections:
    pattern = os.path.join(DATA_DIR, 'raw', section, 'wsj_*')
    all_filepaths.extend(glob(pattern))
random.shuffle(all_filepaths)

# Count current lines
total_sentences = 0
current_paths = glob(os.path.join(OUTPUT_DIR, 'wsj*.txt'))
for path in current_paths:
    total_sentences += len([l for l in open(path, 'r').readlines()
                            if l.strip()]) # don't count blank lines
print 'Existing corpus:', total_sentences, 'sentences'

if total_sentences > MAX_SENTENCES:
    sys.exit()

existing_paths = (glob(os.path.join(OUTPUT_DIR, 'skip', 'wsj*.txt')) +
                  glob(os.path.join(OUTPUT_DIR, 'wsj*.txt')))
existing_basenames = set(os.path.basename(path) for path in existing_paths)

# Copy over files until we've filled our quota
for filepath in all_filepaths:
    basename = os.path.basename(filepath)
    if basename in existing_basenames:
        continue

    mrg_filename = basename + '.mrg'
    mrg_filepath = os.path.join(DATA_DIR, mrg_filename)
    if not os.path.exists(mrg_filepath):
        print 'Skipping', filepath
        continue

    print "Processing", filepath
    with open(filepath, 'r') as in_file:
        lines = in_file.readlines()
    assert lines[:2] == ['.START \n', '\n']
    content_lines = lines[2:]
    with open(os.path.join(OUTPUT_DIR, basename + '.txt'), 'w') as out_file:
        out_file.writelines(content_lines)

    shutil.copy(mrg_filepath, os.path.join(OUTPUT_DIR, mrg_filename))

    total_sentences += len([l for l in content_lines if l.strip()]) # don't count blank lines
    if total_sentences > 1400:
        break
