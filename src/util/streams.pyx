"""
A module of stream-reading utilities that get sped up noticeably when compiled.

To compile:
cython -a streams.pyx
gcc -shared -pthread -fPIC -fwrapv -O2 -Wall -fno-strict-aliasing \
    -I/usr/include/python2.7 -o streams.so streams.c
"""

import io
import struct
from types import MethodType

cdef extern from "ctype.h":
    int tolower( int )

class CharacterTrackingStreamWrapper(io.TextIOWrapper):
    '''
    In encoded files, tell() gives byte offsets. We want character offsets, so
    we have to track them ourselves.
    '''
    def __init__(self, *args, **kwargs):
        super(CharacterTrackingStreamWrapper, self).__init__(*args, **kwargs)
        self.character_position = 0

    def tell(self):
        '''
        The state of the file wrapper now includes a character count. We have
        to make sure that passing the value of tell() back to seek() will
        correctly reproduce the state of the file-like object.
        '''
        # Return a single number encoding a packed list of two integers.
        packed = struct.pack(
            '=LL', super(CharacterTrackingStreamWrapper, self).tell(),
            self.character_position)
        return struct.unpack('=Q', packed)[0]

    def seek(self, long offset, short whence=io.SEEK_SET):
        if whence == io.SEEK_SET:
            if offset == 0:
                super(CharacterTrackingStreamWrapper, self).seek(0, whence)
                self.character_position = 0
            else:
                repacked = struct.pack('=Q', offset)
                textio_tell, character_position = struct.unpack('=LL', repacked)
                super(CharacterTrackingStreamWrapper, self).seek(textio_tell,
                                                                 whence)
                self.character_position = character_position
        elif whence == io.SEEK_END and offset == 0:
            # read() implementation will leave us at the end, and also count
            # relevant characters.
            self.read()
        else:
            super(CharacterTrackingStreamWrapper, self).seek(offset, whence)

    @classmethod
    def __add_overriden_read__(cls, str method_name):
        # This should really be using super(), but that doesn't seem to work
        # well with getattr unless we do it in read_fn, where we have access to
        # self. But then it's slow.
        super_method = getattr(io.TextIOWrapper, method_name)
        def read_fn(self, *args, **kwargs):
            # print "Executing", method_name
            data = super_method(self, *args, **kwargs)
            self.character_position += len(data)
            return data
        new_method = MethodType(read_fn, None, cls)
        setattr(cls, method_name, new_method)

    @classmethod
    def __add_overriden_write__(cls, method_name):
        def write_fn(self, *args, **kwargs):
            raise IOError("Writing is not supported for %s" % cls.__name__)
        new_method = MethodType(write_fn, None, cls)
        setattr(cls, method_name, new_method)

    def __iter__(self):
        try:
            while True:
                line = self.next()
                yield line
        except StopIteration:
            pass

    def __repr__(self):
        return '<%s wrapping %s>' % (type(self).__name__,
                                     super(CharacterTrackingStreamWrapper,
                                           self).__repr__())

# readlines() and next() just call these two.
for name in ['read', 'readline']:
    CharacterTrackingStreamWrapper.__add_overriden_read__(name)

for name in ['write', 'writeline', 'writelines', '__reduce__',
             '__reduce_ex__']:
    CharacterTrackingStreamWrapper.__add_overriden_write__(name)


def eat_whitespace(stream, return_ws=False):
    ''' NOTE: Does not work for interactive streams. '''
    # Skip leading ws
    ws = (None, '')[return_ws]
    while True:
        saved = stream.tell()
        next_char = stream.read(1)
        if next_char == '': # we've hit EOF
            break
        elif next_char.isspace():
            if return_ws:
                ws += next_char
        else: # We got a character, but it wasn't a space. Revert.
            stream.seek(saved)
            break

    return ws

def is_at_eof(stream):
    ''' NOTE: Does not work for interactive streams. '''
    position = stream.tell()
    if stream.read(1) == '':
        return True
    else:
        stream.seek(position)
        return False

def peek_and_revert_unless(stream, condition=None):
    position = stream.tell()
    next_char = stream.read(1)
    test_result = condition is None or condition(next_char)
    if not test_result:
        stream.seek(position)
    return next_char, test_result

def read_stream_until(stream, basestring delimiter,
                      bint case_insensitive=False):
    ''' NOTE: Does not work for interactive streams. '''
    cdef basestring accumulator, lower_delimiter, next_char
    cdef long found_delimiter_until

    if len(delimiter) == 0:
        return (u'', True)
    lower_delimiter = delimiter.lower()

    accumulator = u''
    found_delimiter_until = 0  # index *after* last delim character found
    while True:
        next_char = stream.read(1)
        if len(next_char) == 0: # we've hit EOF
            break

        accumulator += next_char
        # found_delimiter_until < len(delimiter), so what follows is safe.
        if next_char == delimiter[found_delimiter_until] or (
                case_insensitive and tolower(ord(next_char[0])) ==
                lower_delimiter[found_delimiter_until]):
            found_delimiter_until += 1
            if found_delimiter_until == len(delimiter):
                return (accumulator, True)
        else: # next delimiter character not fonud
            found_delimiter_until = 0

    return (accumulator, False)
