from __future__ import print_function
from itertools import tee, izip
import os

def recursively_list_files(path):
    walker = os.walk(path)
    while True:
        try:
            root, dirs, files = walker.next()
            dirs.sort()
            files.sort()
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
    if not dictionaries:
        return {}

    d = dictionaries[0].copy()
    for next_dict in dictionaries[1:]:
        d.update(next_dict)
    return d

def truncated_string(string, truncate_to=25):
    truncated = string[:truncate_to]
    if len(truncated) < len(string):
        truncated += '...'
    return truncated

def partition(list_to_partition, num_partitions):
    '''
    Returns a list of lists, dividing list_to_partition as evenly as possible
    into sublists.
    From http://stackoverflow.com/a/2660034/4044809
    '''
    division = len(list_to_partition) / float(num_partitions)
    partitions = [list_to_partition[int(round(division * i)):
                                    int(round(division * (i + 1)))]
                  for i in xrange(num_partitions)]
    return partitions

def print_indented(indent_level, *args, **kwargs):
    single_indent_str = kwargs.pop('single_indent_str', '    ')

    if indent_level == 0:
        print(*args, **kwargs)
    else:
        prefix = single_indent_str * indent_level
        prefix_kwargs = dict(kwargs)
        prefix_kwargs['end'] = ''

        stringified = [str(arg) for arg in args]
        result_str = kwargs.get('sep', ' ').join(stringified)
        for line in result_str.split('\n'):
            print(prefix, **prefix_kwargs)
            print(line, **kwargs)

# From https://docs.python.org/2/library/itertools.html
def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return izip(a, b)

# From http://stackoverflow.com/a/3010495/4044809
def get_terminal_size():
    import fcntl, termios, struct
    h, w, _, _ = struct.unpack('HHHH',
        fcntl.ioctl(0, termios.TIOCGWINSZ,
        struct.pack('HHHH', 0, 0, 0, 0)))
    return w, h
