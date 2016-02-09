from __future__ import print_function

import colorama
import fcntl
from itertools import tee, izip, izip_longest
import numpy as np
import os
import struct
import termios


class NameDictionary(object):
    def __init__(self):
        self.names_to_ids = {}
        self.ids_to_names = []

    def insert(self, entry):
        if not self.names_to_ids.has_key(entry):
            self.names_to_ids[entry] = len(self.names_to_ids)
            self.ids_to_names.append(entry)

    def clear(self):
        self.__init__()

    def __getitem__(self, entry):
        if isinstance(entry, int):
            return self.ids_to_names[entry]
        else: # it's a string name
            return self.names_to_ids[entry]

    def __len__(self):
        return len(self.names_to_ids)

    def __contains__(self, entry):
        return self.names_to_ids.has_key(entry)


# Add some Colorama functionality.
class AnsiFormats:
    BOLD = 1
    UNDERLINE = 4
    BLINK = 5
    RESET_ALL = 0
colorama.Format = colorama.ansi.AnsiCodes(AnsiFormats)

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

class Enum(list): # based on http://stackoverflow.com/a/9201329 (but faster)
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

def partition(list_to_partition, num_partitions, item_weights=None,
              allow_empty_partitions=False):
    '''
    Returns a list of lists, dividing list_to_partition as evenly as possible
    into sublists. item_weights, if provided, are assumed to be positive, and
    must be the same size as list_to_partition.
    Based on http://stackoverflow.com/a/2660034/4044809.
    '''
    if num_partitions > len(list_to_partition) and not allow_empty_partitions:
        raise ValueError("Can't partition {} items into {} partitions".format(
            len(list_to_partition), num_partitions))
    # TODO: replaced weighted version with a cleverer algorithm?
    if item_weights:
        assert len(item_weights) == len(list_to_partition)
        division = sum(item_weights) / float(num_partitions)
        partitions = [[] for _ in range(num_partitions)]
        partition_weight = 0
        partition_iter = iter(partitions)
        partition = partition_iter.next()
        for item, weight in zip(list_to_partition, item_weights):
            partition.append(item)
            partition_weight += weight
            if (partition_weight + weight > division
                and partition is not partitions[-1]):
                partition_weight = 0
                partition = partition_iter.next()

        '''
        try:
            remaining = partition_iter.next()
            assert False
        except StopIteration:
            pass
        '''

    else:
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
    h, w, _, _ = struct.unpack('HHHH',
        fcntl.ioctl(0, termios.TIOCGWINSZ,
        struct.pack('HHHH', 0, 0, 0, 0)))
    return w, h

# From http://stackoverflow.com/a/434411
def igroup(iterable, n, fillvalue=None):
    args = [iter(iterable)] * n
    return izip_longest(*args, fillvalue=fillvalue)

def floats_same_or_nearly_equal(f1, f2):
    '''
    NaN != NaN. This function compares floats in a way that considers them equal
    if they are both NaNs OR they are actually equal. Useful for testing.
    '''
    if np.isnan(f1):
        return np.isnan(f2)
    else:
        return np.allclose(f1, f2)

class Object(object):
    ''' Dummy container class for storing attributes in. '''
    pass
