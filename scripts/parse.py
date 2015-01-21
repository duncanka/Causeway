import os
import subprocess
from tempfile import NamedTemporaryFile

def run_parser(sentence, parser_path, write_results=False):
    if write_results:
        delete = False
        output_format = 'words,wordsAndTags,typedDependencies'
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
        parse_path = os.path.splitext(sentence_file.name)[0] + '.parse'
        with open(parse_path, 'w') as parse_file:
            parse_file.write(stdout)
        return (sentence_file.name, parse_path)
    else:
        return stdout

def get_parsed_sentence(sentence_text):
    txt_path, _ = run_parser(
        sentence_text, '../../../stanford-parser-full-2014-10-31/', True)
    from data.readers import SentenceReader
    r = SentenceReader()
    r.open(txt_path)
    sentence = r.get_next()
    r.close()
    return sentence
