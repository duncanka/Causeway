from __future__ import print_function

import colorama
import fcntl
import hashlib
import importlib
from itertools import chain, combinations, izip, izip_longest, tee
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

    # Pickling functions (to improve efficiency)

    def __getstate__(self):
        return self.ids_to_names

    def __setstate__(self, ids_to_names):
        self.ids_to_names = ids_to_names
        self.names_to_ids = {}
        for i, name in enumerate(ids_to_names):
            self.names_to_ids[name] = i

# Add some Colorama functionality.
for style, code in [('UNDERLINE', 4), ('BLINK', 5)]:
    setattr(colorama.Style, style, '\033[%dm' % code)


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

    def __repr__(self, *args, **kwargs):
        return 'Enum(%s)' % list.__repr__(self, *args, **kwargs)


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
        for line in result_str.split(os.linesep):
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


def get_object_by_fqname(fqname):
    '''
    fqname must be a string starting with a module name.
    '''
    splits = fqname.split('.')
    for i in range(len(splits), 0, -1):
        module_name = '.'.join(splits[:i])
        try:
            module = importlib.import_module(module_name)
        except:
            continue

        obj = module
        for j in range (i, len(splits)):
            obj = getattr(obj, splits[j])
        return obj

    raise AttributeError('No such object: %s' % fqname)


class Object(object):
    ''' Dummy container class for storing attributes in. '''
    pass


def make_getter(underlying_attr_name):
    def getter(self):
        return getattr(self, underlying_attr_name)
    return getter


def make_setter(underlying_attr_name):
    def setter(self, value):
        return setattr(self, underlying_attr_name, value)
    return setter


def hash_file(filename, chunk_size=1024 * 1024):
    """" This function returns the SHA-256 hash
         of the file passed into it. """
    h = hashlib.sha256()

    with open(filename, 'rb') as in_file:
        chunk = 0
        while chunk != b'':
            chunk = in_file.read(chunk_size)
            h.update(chunk)

    return h.hexdigest()


# From https://docs.python.org/2/library/itertools.html#recipes
def powerset(iterable):
    xs = list(iterable)
    # note we return an iterator rather than a list
    return chain.from_iterable(combinations(xs, n) for n in range(len(xs) + 1))


class MinMaxRange(object):
    """
    Simple class to track the min and max of a series of observed values.
    """
    def __init__(self, initial_min=np.inf, initial_max=-np.inf,
                 initial_values=None):
        self.min = initial_min
        self.max = initial_max
        if initial_values is not None:
            self.update(initial_values)

    def update(self, values):
        try:
            self.min = min(self.min, np.min(values))
            self.max = max(self.max, np.max(values))
        except ValueError: # Happens with zero-size values
            pass

    def __repr__(self):
        return 'MinMaxRange(%f, %f)' % (self.min, self.max)
