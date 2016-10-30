from collections import defaultdict
from itertools import chain
import logging

from causality_pipelines.tregex_based.tregex_stage import TRegexConnectiveModel
from data.io import DirectoryReader, CausalityStandoffReader
from iaa import stringify_connective
from util import print_indented


if __name__ == '__main__':
    logging.basicConfig(
        format='%(filename)s:%(lineno)s:%(levelname)s: %(message)s',
        level=logging.INFO)
    logging.captureWarnings(True)

    reader = DirectoryReader((CausalityStandoffReader.FILE_PATTERN,),
                             CausalityStandoffReader())

    reader.open("/var/www/brat/data/finished")
    all_docs = reader.get_all()
    all_instances = chain.from_iterable(
        s.causation_instances for s in
        chain.from_iterable(d.sentences for d in all_docs))
    patterns_and_instances_by_connective = defaultdict(lambda: defaultdict(list))
    to_remap = {'for too to': 'too for to', 'for too': 'too for',
                'reason be': 'reason', 'that now': 'now that',
                'to for': 'for to', 'give': 'given', 'result of': 'result'}

    for i in all_instances:
        if not i.cause or not i.effect:
            continue

        pattern, _node_names = TRegexConnectiveModel._get_dependency_pattern(
            i.sentence, i.connective, i.cause, i.effect)
        if not pattern:
            continue

        connective = stringify_connective(i)
        if connective.startswith('be '):
            connective = connective[3:]
            # print "Replaced", connective
        elif connective in to_remap:
            connective = to_remap[connective]
            # print 'Replaced', connective
        patterns_and_instances_by_connective[connective][pattern].append(i)
        
    for connective, by_pattern in patterns_and_instances_by_connective.iteritems():
        print '"%s":' % connective
        print_indented(1, len(by_pattern), 'unique patterns')
        for pattern, instances in by_pattern.iteritems():
            print_indented(1, pattern.encode('utf-8'))
            print
            for instance in instances:
                print_indented(2, str(instance.sentence))
                print
            print
