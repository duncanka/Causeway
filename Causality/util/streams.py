def eat_whitespace(stream, return_ws=False):
    # Skip leading ws
    ws = (None, '')[return_ws]
    for next_char in iter(lambda: stream.read(1), ''):
        if next_char.isspace():
            if return_ws:
                ws += next_char
        else:
            # Seek back 1 character from the current position to the first
            # non-whitespace character.
            stream.seek(-1, 1)
            break
    return ws

def is_at_eof(stream):
    position = stream.tell()
    try:
        stream.next()
        stream.seek(position)
        return False
    except StopIteration:
        return True

def peek_and_revert_unless(stream, condition=lambda char: True):
    position = stream.tell()
    next_char = stream.read(1)
    test_result = condition(next_char)
    if not test_result:
        stream.seek(position)
    return next_char, test_result

def read_stream_until(stream, delimiter, case_insensitive, accumulate=True):
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
