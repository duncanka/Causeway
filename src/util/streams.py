import io
import struct
from types import MethodType

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

    def seek(self, offset, whence=io.SEEK_SET):
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
    def __add_overriden_read__(cls, method_name):
        def read_fn(self, *args, **kwargs):
            # print "Executing", method_name
            super_method = getattr(super(cls, self), method_name)
            data = super_method(*args, **kwargs)
            if isinstance(data, list):
                self.character_position += sum(len(item) for item in data)
            else:
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

for name in ['read', 'readline']: # readlines() and next() just call these two
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

def peek_and_revert_unless(stream, condition=lambda char: True):
    position = stream.tell()
    next_char = stream.read(1)
    test_result = condition(next_char)
    if not test_result:
        stream.seek(position)
    return next_char, test_result

def read_stream_until(stream, delimiter, case_insensitive=False,
                      accumulate=True):
    ''' NOTE: Does not work for interactive streams. '''
    accumulator = (None, '')[accumulate]

    if not delimiter:
        return (accumulator, True)

    found_delimiter_until = 0  # index *after* last delim character found
    for next_char in iter(lambda: stream.read(1), ''):
        if accumulate:
            accumulator += next_char
        # found_delimiter_until < len(delimiter), so what follows is safe.
        if next_char == delimiter[found_delimiter_until] or (
                case_insensitive and next_char.lower() ==
                delimiter[found_delimiter_until].lower()):
            found_delimiter_until += 1
            if found_delimiter_until == len(delimiter):
                return (accumulator, True)
        else: # next delimiter character not fonud
            found_delimiter_until = 0

    return (accumulator, False)
