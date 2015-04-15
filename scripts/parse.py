import os
import subprocess
from tempfile import NamedTemporaryFile

def write_parse_results(txt_file_name, parse_text):
    parse_path = os.path.splitext(txt_file_name)[0] + '.parse'
    with open(parse_path, 'w') as parse_file:
        parse_file.write(parse_text)
    return parse_path

def run_parser(sentence, parser_path, write_results=False):
    if write_results:
        delete = False
        output_format = 'words,wordsAndTags,penn,typedDependencies'
    else:
        delete = True
        output_format = 'typedDependencies'

    with NamedTemporaryFile(suffix='.txt', delete=delete) as sentence_file:
        sentence_file.write(sentence)
        sentence_file.file.flush()
        parser_process = subprocess.Popen(
            ['java', '-mx4600m', '-cp', '%s/*' % parser_path,
             'edu.stanford.nlp.parser.lexparser.LexicalizedParser',
             '-tokenizerOptions',
             'ptb3Escaping=false,americanize=false,untokenizable=firstKeep',
             '-outputFormat', output_format,
             '-outputFormatOptions', 'nonCollapsedDependencies,stem',
             'edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz',
             sentence_file.name],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, _ = parser_process.communicate()

    if write_results:
        parse_path = write_parse_results(sentence_file.name, stdout)
        return (sentence_file.name, parse_path)
    else:
        return stdout

def get_parsed_sentence(sentence_text, parse_text=None):
    if parse_text:
        with NamedTemporaryFile(suffix='.txt', delete=False) as sentence_file:
            sentence_file.write(sentence_text)
        txt_path = sentence_file.name
        parse_path = write_parse_results(sentence_file.name, parse_text)
    else:
        txt_path, parse_path = run_parser(
            sentence_text, '../../../stanford-parser/', True)

    from data.readers import SentenceReader
    r = SentenceReader()
    r.open(txt_path)
    sentence = r.get_next()
    r.close()

    os.remove(parse_path)
    os.remove(txt_path)

    return sentence
