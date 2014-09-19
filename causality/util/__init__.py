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

class Enum(tuple): # based on http://stackoverflow.com/a/9201329 (but faster)
    def __init__(self, names):
        super(Enum, self).__init__(names)
        for i, name in enumerate(names):
            setattr(self, name, i)

def listify(arg):
    """Wraps arg in a list if it's not already a list or tuple."""
    if isinstance(arg, list) or isinstance(arg, tuple):
        return arg
    return [arg]

def merge_dicts(dictionaries):
    d = dictionaries[0]
    for next_dict in dictionaries[1:]:
        d.update(next_dict)
    return d
