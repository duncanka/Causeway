from itertools import chain

from data.io import CausalityStandoffReader, DirectoryReader

def get_reader():
    return DirectoryReader((CausalityStandoffReader.FILE_PATTERN,),
                           CausalityStandoffReader())

def read_all(datadir='/var/www/brat/data/finished'):
    reader = get_reader()
    reader.open(datadir)
    all_sentences = chain.from_iterable(d.sentences for d in reader)
    all_instances = chain.from_iterable(s.causation_instances
                                        for s in all_sentences)
    all_instances = list(all_instances)
    reader.close()
    return all_instances
