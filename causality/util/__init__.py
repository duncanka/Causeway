import os

def recursively_list_files(path):
    walker = os.walk(path)
    while True:
        try:
            root, dirs, files = walker.next()
            for filename in files:
                yield os.path.join(root, filename)
        except StopIteration:
            break

class Enum(tuple): # based on http://stackoverflow.com/a/9201329
    __getattr__ = tuple.index

def listify(arg):
    """Wraps arg in a list if it's not already a list or tuple."""
    if isinstance(arg, list) or isinstance(arg, tuple):
        return arg
    return [arg]
