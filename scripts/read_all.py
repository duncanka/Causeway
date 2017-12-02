from itertools import chain

from causeway.because_data import CausalityStandoffReader
from nlpypline.data.io import DirectoryReader


def get_reader(recursive=False):
    return DirectoryReader((CausalityStandoffReader.FILE_PATTERN,),
                           CausalityStandoffReader(), recursive)


def docs_to_instances(docs, overlapping=False):
    all_sentences = chain.from_iterable(d.sentences for d in docs)
    if overlapping:
        all_instances = chain.from_iterable(s.overlapping_rel_instances
                                            for s in all_sentences)
    else:
        all_instances = chain.from_iterable(s.causation_instances
                                            for s in all_sentences)
    return list(all_instances)


def read_all(datadir='/var/www/brat/data/finished', instances=True,
             overlapping=False, recursive=False):
    reader = get_reader(recursive)
    reader.open(datadir)

    if instances:
        all_instances = docs_to_instances(reader, overlapping)
    else:
        all_instances = reader.get_all()

    reader.close()
    return all_instances
