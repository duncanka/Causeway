#!/usr/bin/env python

from nltk.corpus.reader import XMLCorpusReader
import os
import sys
from collections import defaultdict
#from multiprocessing import Process, Queue, cpu_count

### Functions to create extractors ###

def make_filter_extractor(base_extractor, reject_cond, feature_name,
                          allow_blank):
    def extractor(elt):
        contents = base_extractor(elt)
        if contents is None or not reject_cond(contents):
            return contents
        else:
            if allow_blank:
                return None
            else:
                raise ValueError(feature_name)
    return extractor

def make_attr_extractor(attr_name):
    return lambda elt: elt.get(attr_name)

def make_class_text_extractor(class_name):
    def text_if_online_class(elt):
        if elt.get('class') != class_name:
            return None
        return elt.text
    return text_if_online_class

def make_body_extractor(kicker):
    if len(kicker) == 0:
        return lambda elt: '\n\n'.join([c.text for c in elt.getchildren()])
    else:
        kicker = kicker.__iter__().next()
        def extractor(elt):
            pars = [c.text for c in elt.getchildren()]
            if pars[-1].strip() == kicker:
                pars.pop()
            return '\n\n'.join(pars)
        return extractor

### Global extractors ###

text_extractor = lambda elt: elt.text
content_extractor = make_attr_extractor('content')
docid_extractor = make_attr_extractor('id-string')
kicker_extractor = make_attr_extractor('series.name')
indexing_extractor = make_class_text_extractor('indexing_service')
online_section_extractor = make_filter_extractor(
    lambda elt: [sec.strip() for sec in elt.get('content').split(';')],
    lambda sections: (('Washington' not in sections)
                      or 'Corrections' in sections),
    'online_section', False)
descriptor_extractor = make_filter_extractor(
    indexing_extractor, lambda contents: contents.startswith("NO INDEX TERMS"),
    "descriptor", True)
mat_type_extractor = make_filter_extractor(
    text_extractor, lambda contents: contents in (
        "Summary", "List", "Paid Death Notice", "Paid Memorial Notice"),
    "material_type", False)

### Generic functions for running extractors on the document and handling the results ###

def extract_feature(doc, xpath, extractor=text_extractor):
    result = set()
    elts = doc.findall(xpath)
    for elt in elts:
        extracted = extractor(elt)
        if extracted is None:
            continue
        # Extractor can return multiple items. If it did, add them all.
        if hasattr(extracted, '__iter__'):
            result.update(extracted)
        else:
            result.add(extracted)

    return result

def extract_taxonomies(doc):
    tax_classes = doc.findall(
        './/head/docdata/identified-content/classifier'
        '[@type="taxonomic_classifier"]')
    unique_classes = set()
    # Shave off first 4 chars, because they'll just be "Top/"
    tax_classes = [c.text[4:] for c in tax_classes]
    for tax_class in tax_classes:
        classes_to_del = set()
        add_to_unique = True
        for c in unique_classes:
            if c.startswith(tax_class):
                add_to_unique = False
                break # tax_class is the same as or a prefix of something we've seen already -- ignore it
            elif tax_class.startswith(c):
                # c is a prefix of this next class, so we should delete c later
                classes_to_del.add(c)
        unique_classes = unique_classes - classes_to_del
        if add_to_unique:
            unique_classes.add(tax_class)

    return unique_classes

def add_features(name, doc_dict, values, max_allowed=1, required=True):
    if len(values) > max_allowed:
        raise ValueError(name)
    elif len(values) == 0:
        if required:
            raise ValueError(name)

    for i, value in enumerate(values):
        doc_dict[name + ("_%d" % i)] = value
    for i in range(len(values), max_allowed):
        doc_dict[name + ("_%d" % i)] = ''


### Driver ###

sections = set()

def process_doc(doc):
    doc_dict = {}

    '''
    add_features('doc_id', doc_dict,
                 extract_feature(doc, './/head/docdata/doc-id', docid_extractor))
    add_features('headline', doc_dict,
                 extract_feature(doc, './/body[1]/body.head/hedline/hl1'))
    add_features('publication_year', doc_dict,
                 extract_feature(doc, './/head/meta[@name="publication_year"]', content_extractor))
    add_features('publication_month', doc_dict,
                 extract_feature(doc, './/head/meta[@name="publication_month"]', content_extractor))
    add_features('taxonomic_labels', doc_dict, extract_taxonomies(doc), 9)
    '''
    kicker = extract_feature(doc, './/head/docdata/series', kicker_extractor)
    add_features('body', doc_dict,
                 extract_feature(doc, './/body/body.content/block[@class="full_text"]',
                                 make_body_extractor(kicker)))
    add_features('material_type', doc_dict,
                 extract_feature(doc, './/head/docdata/identified-content/classifier[@type="types_of_material"]',
                                 mat_type_extractor),
                 4, False)

    #add_features('day_of_week', doc_dict,
    #             extract_feature(doc, './/head/meta[@name="publication_day_of_week"]', content_extractor))
    #add_features('descriptor', doc_dict,
    #             extract_feature(doc, './/head/docdata/identified-content/classifier[@type="descriptor"]',
    #                             descriptor_extractor),
    #             8, False)
    #add_features('general_descriptor', doc_dict,
    #             extract_feature(doc, './/head/docdata/identified-content/classifier[@type="general_descriptor"]'),
    #             10, False)
    #add_features('news_desk', doc_dict,
    #             extract_feature(doc, './/head/meta[@name="dsk"]', content_extractor))
    add_features('online_section', doc_dict,
                 extract_feature(doc, './/head/meta[@name="online_sections"]',
                                 online_section_extractor),
                 4)
    sections.update([doc_dict[x] for x in doc_dict if x.startswith('online_section')])
    #add_features('print_section', doc_dict,
    #             extract_feature(doc, './/head/meta[@name="print_section"]', content_extractor))
    #add_features('print_section', doc_dict,
    #             extract_feature(doc, './/head/meta[@name="print_section"]', content_extractor))

    #add_features('kicker', doc_dict, kicker, required=False)

    return doc_dict

def doc_path_to_dict(path):
    directory, fname = os.path.split(path)
    reader = XMLCorpusReader(directory, fname)
    doc = reader.xml()
    try:
        return process_doc(doc)
    except ValueError, e:
        return e.args[0]

def worker(input, output):
    for path in iter(input.get, 'STOP'):
        output.put((path, doc_path_to_dict(path)))

def main(argv):
    root_path = argv[1]
    target_path = argv[2] if len(argv) > 2 else None

    file_paths = []
    for dirpath, _dirs, files in os.walk(root_path, topdown=False):
        file_paths.extend([os.path.join(dirpath, filename) for filename in files
                           if filename.endswith('.xml')])
    num_paths = len(file_paths)
    print "Found", num_paths, "files"
    skipped = defaultdict(int)

    class Dummy(object): pass
    num_done = Dummy()
    num_done.val = 0
    def handle_result(path, doc_dict):
        if isinstance(doc_dict, str):
            skipped[doc_dict] += 1
        else:
            dir_path, filename = os.path.split(path)
            if target_path:
                dir_path = target_path
            path_base = os.path.join(dir_path, os.path.splitext(filename)[0])
            with open(path_base + '.txt', 'w') as txt_file:
                txt_file.write(doc_dict['body_0'].encode('utf-8'))
            open(path_base + '.ann', 'a').close() # create empty .ann file
        num_done.val += 1
        sys.stdout.write('\r%d / %d' % (num_done.val, num_paths))

    '''
    path_q = Queue()
    result_q = Queue()

    for i in range(cpu_count()):
        Process(target=worker, args=(path_q, result_q)).start()
    for path in file_paths:
        path_q.put(path)
    path_q.put('STOP')

    while not path_q.empty() or not result_q.empty():
        handle_result(*result_q.get())
    '''

    for path in file_paths:
        handle_result(path, doc_path_to_dict(path))

    sys.stdout.write("\n")
    print sections
    print "Skipped:", dict(skipped)
    print "Total skipped:", sum(skipped.values())

if __name__ == '__main__':
    main(sys.argv)
