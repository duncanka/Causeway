from itertools import chain

from data.io import CausalityStandoffReader, DirectoryReader

def read_all(datadir='/var/www/brat/data/finished'):
    reader = DirectoryReader((CausalityStandoffReader.FILE_PATTERN,),
                             CausalityStandoffReader())
    reader.open(datadir)
    all_docs = reader.get_all()
    all_sentences = chain.from_iterable(d.sentences for d in all_docs)
    all_instances = chain.from_iterable(s.causation_instances
                                        for s in all_sentences)
    return list(all_instances)
