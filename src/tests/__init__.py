from os import path

def get_resources_dir():
    return path.join(path.dirname(__file__), 'resources')
