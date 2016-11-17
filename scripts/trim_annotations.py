from gflags import FLAGS, DEFINE_integer, DEFINE_string, DuplicateFlagError
import io
from os import path
import shutil
import sys

from causeway.because_data import CausalityStandoffReader, CausalityStandoffWriter
from data.io import StanfordParsedSentenceReader


try:
    DEFINE_integer('start_sentence', 0, 'Sentence at which to start copying')
    DEFINE_integer('end_sentence', -1, 'Sentence at which to stop copying')
    DEFINE_string('out_file_name', None,
                  'Base name for output file. Defaults to same as input file.')
except DuplicateFlagError:
    pass


if __name__ == '__main__':
    argv = sys.argv
    argv = FLAGS(argv)
    in_file, out_directory = argv[1:]

    reader = CausalityStandoffReader(in_file)
    doc = reader.get_next()

    if FLAGS.out_file_name is None:
        base_name = path.splitext(path.split(in_file)[1])[0]
    else:
        base_name = FLAGS.out_file_name

    in_txt_name = path.splitext(in_file)[0] + '.txt'
    out_txt_name = path.join(out_directory, base_name + '.txt')
    out_parse_name = path.join(out_directory, base_name + '.parse')

    if FLAGS.start_sentence > 0:
        start_copying_char = doc.sentences[
            FLAGS.start_sentence].document_char_offset
    else:
        start_copying_char = 0

    sentence_reader = StanfordParsedSentenceReader(in_txt_name)
    for i in range(FLAGS.start_sentence):
        sentence_reader.get_next_sentence()

    with io.open(out_txt_name, 'wb') as out_txt_file:
        in_txt_file = sentence_reader._file_stream
        while in_txt_file.character_position < start_copying_char:
            in_txt_file.read(1)
        shutil.copyfileobj(in_txt_file, out_txt_file)
    with io.open(out_parse_name, 'wb') as out_parse_file:
        shutil.copyfileobj(sentence_reader._parse_file, out_parse_file)

    out_ann_name = path.join(out_directory, base_name + '.ann')
    writer = CausalityStandoffWriter(out_ann_name, start_copying_char)

    def instances_getter(document):
        if FLAGS.end_sentence == -1:
            sentences_to_write = document.sentences[FLAGS.start_sentence:]
        else:
            sentences_to_write = document.sentences[FLAGS.start_sentence:
                                                    FLAGS.end_sentence + 1]
        return sentences_to_write
    writer.write_all_instances(doc, instances_getter)
