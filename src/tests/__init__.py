from os import path

def get_resources_dir():
    return path.join(path.dirname(__file__), 'resources')

def get_sentences_from_file(reader_type, subdir, filename):
        reader = reader_type()
        reader.open(path.join(get_resources_dir(), subdir, filename))
        sentences = reader.get_all()
        reader.close()
        return sentences
