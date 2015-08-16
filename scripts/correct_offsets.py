#!/usr/bin/env python
# For correcting annotation offsets after quotes have been converted

import glob
import os
import sys

def main(argv):
    originals_directory = argv[1]
    ann_directory = argv[2]
    txt_files = glob.glob(originals_directory + "/*.txt")
    for txt_file_name in txt_files:
        process_file(txt_file_name, ann_directory)

def process_file(txt_file_name, ann_directory):
    with open(txt_file_name, 'r') as txt_file:
        file_txt = txt_file.read()
    quote_indices = [i for i, ltr in enumerate(file_txt) if ltr == '"']

    ann_file_name = os.path.basename(os.path.splitext(txt_file_name)[0]) + ".ann"
    ann_file_name = os.path.join(ann_directory, ann_file_name)
    with open(ann_file_name, 'r') as ann_file:
        ann_file_lines = ann_file.readlines()
    with open(ann_file_name, 'w') as ann_file:
    # with open('/dev/null', 'r') as thing:
        for line in ann_file_lines:
            if line[0] == 'T':
                line = correct_line(line, quote_indices)
            ann_file.write(line)

def correct_line(line, quote_indices):
    id, annotation, text = line.split('\t')

    annotation_segments = annotation.split(' ')
    new_annotation_segments = [annotation_segments[0]]
    for segment in annotation_segments[1:]:
        semicolon_pieces = segment.split(';')
        new_semicolon_pieces = []
        for semicolon_piece in semicolon_pieces:
            index = int(semicolon_piece)
            to_add = 0
            for quote_index in quote_indices:
                if quote_index < index:
                    to_add += 1
            index += to_add
            new_semicolon_pieces.append(str(index))
        new_annotation_segments.append(';'.join(new_semicolon_pieces))

    new_line = '\t'.join([id, ' '.join(new_annotation_segments), text])
    print 'Corrected', repr(line), 'to', repr(new_line)
    return new_line

if __name__ == '__main__':
    main(sys.argv)
