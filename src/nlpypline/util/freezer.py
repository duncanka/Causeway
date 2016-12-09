'''
Defines functionality for freezing any Python object -- i.e., making it
immutable. Of course, Python doesn't allow actually making an object immutable;
the functionality defined here will only make an immutable reference to it. If
recursive is set to True, attributes and elements of the object will also be
immutable when accessed through this reference.
'''

import types

class FrozenError(TypeError): pass

class Frozen(object):
    immutable_types = [] # Set for real below

    @staticmethod
    def _freeze_if_needed(obj, recursive):
        if not recursive:
            return obj
        elif type(obj) in Frozen.immutable_types:
            return obj
        else:
            return freeze(obj, recursive)

    def __init__(self, value, recursive=True):
        '''
        `value`: the object to wrap.
        `recursive`: whether all attributes, members, etc. accessed through
        '''
        # Circumvent our overridden __setattr__ function.
        object.__setattr__(self, '_value', value)
        object.__setattr__(self, '_recursive', recursive)

    ### Accessors ###

    def __getattribute__(self, name):
        _value = object.__getattribute__(self, '_value')
        _recursive = object.__getattribute__(self, '_recursive')
        attr = getattr(_value, name)
        return Frozen._freeze_if_needed(attr, _recursive)

    def __getitem__(self, i):
        _value = object.__getattribute__(self, '_value')
        try:
            return _value.__getitem__(i)
        except AttributeError: # Assume this was a slice on a built-in object.
            return _value.__getslice__(i.start, i.stop)

    def __iter__(self):
        _value = object.__getattribute__(self, '_value')
        _recursive = object.__getattribute__(self, '_recursive')
        for next_elt in iter(_value):
            yield Frozen._freeze_if_needed(next_elt, _recursive)

    ### Stringification functions ###

    def __str__(self):
        _value = object.__getattribute__(self, '_value')
        return 'Frozen(%s)' % str(_value)

    def __repr__(self):
        _value = object.__getattribute__(self, '_value')
        return 'Frozen(%s)' % repr(_value)

    def __format__(self, formatstr):
        _value = object.__getattribute__(self, '_value')
        return 'Frozen(%s)' % _value.__class__.__format__(_value, formatstr)

    def __unicode__(self):
        _value = object.__getattribute__(self, '_value')
        return u'Frozen(%s)' % unicode(_value)
    
    ### Misc methods that need special care ###
    def __reversed__(self):
        _value = object.__getattribute__(self, '_value')
        _recursive = object.__getattribute__(self, '_recursive')
        val_reversed = reversed(_value)
        if _recursive:
            val_reversed = freeze(val_reversed)
        return val_reversed
    
    def __coerce__(self, other):
        _value = object.__getattribute__(self, '_value')
        coerced = _value.__coerce__(other)
        if coerced is None:
            return None
        coerced_self, coerced_other = coerced
        _recursive = object.__getattribute__(self, '_recursive')
        coerced_self = self._freeze_if_needed(coerced_self, _recursive)
        return (coerced_self, coerced_other)
    
    def __cmp__(self, other):
        # Some builtins don't have __cmp__ defined, even though they can be used
        # with cmp().
        _value = object.__getattribute__(self, '_value')
        return cmp(_value, other)

    def __eq__(self, other):
        _value = object.__getattribute__(self, '_value')
        if isinstance(other, Frozen):
            other = object.__getattribute__(other, '_value')
        return _value == other

    def __ne__(self, other):
        _value = object.__getattribute__(self, '_value')
        if isinstance(other, Frozen):
            other = object.__getattribute__(other, '_value')
        return _value != other

    def __lt__(self, other):
        _value = object.__getattribute__(self, '_value')
        if isinstance(other, Frozen):
            other = object.__getattribute__(other, '_value')
        return _value < other

    def __gt__(self, other):
        _value = object.__getattribute__(self, '_value')
        if isinstance(other, Frozen):
            other = object.__getattribute__(other, '_value')
        return _value > other

    def __le__(self, other):
        _value = object.__getattribute__(self, '_value')
        if isinstance(other, Frozen):
            other = object.__getattribute__(other, '_value')
        return _value <= other

    def __ge__(self, other):
        _value = object.__getattribute__(self, '_value')
        if isinstance(other, Frozen):
            other = object.__getattribute__(other, '_value')
        return _value >= other

    def __hash__(self, *args, **kwargs):
        _value = object.__getattribute__(self, '_value')
        return hash(_value)

    def __nonzero__(self):
        _value = object.__getattribute__(self, '_value')
        return bool(_value)

    def __call__(self, *args, **kwargs):
        # TODO: calling arbitrary functions is very dangerous. Is there some way
        # we can wrap a call so that the method is tricked into running on a
        # Frozen object, thus allowing only non-mutating operations?
        '''
        _value = object.__getattribute__(self, '_value')
        raise FrozenError("Can't call frozen function {0}".format(_value))
        '''
        _value = object.__getattribute__(self, '_value')
        return _value(*args, **kwargs)
        # '''

    ### Template magic methods for remainder (added for real below) ###

    @staticmethod
    def _illegal_modifier_method(self, *args, **kwargs):
        _value = object.__getattribute__(self, '_value')
        raise FrozenError("Can't modify frozen object {0}".format(_value))

    @staticmethod
    def _get_legal_method(method_name):        
        def _legal_method(self, *args, **kwargs):
            _value = object.__getattribute__(self, '_value')
            underlying_method = getattr(_value, method_name)
            return underlying_method(*args, **kwargs)
        return _legal_method

    @staticmethod
    def _get_reflected_method(method_name):
        def _reflected_method(self, *args, **kwargs):
            _value = object.__getattribute__(self, '_value')
            try:
                underlying_method = getattr(_value, method_name)
            except AttributeError:
                return NotImplemented
            return underlying_method(*args, **kwargs)
        return _reflected_method
    
    @staticmethod
    def _get_unary_method(method_name):
        def _unary_method(self, *args, **kwargs):
            _value = object.__getattribute__(self, '_value')
            underlying_method = getattr(_value, method_name)
            retval = underlying_method(*args, **kwargs)
            if retval is _value:
                return self
            else:
                return retval
        return _unary_method

### Modify the class in ways that can't be done at definition time. ###

Frozen.immutable_types = list(types.StringTypes) + [int, float, bool, complex,
    long, tuple, Frozen, types.NoneType, slice, type, types.NotImplementedType]

# Mutating methods: all illegal.
for method_name in ['setattr', 'setitem', 'delattr', 'delitem', 'iadd', 'isub',
                    'imul', 'ifloordiv', 'idiv', 'itruediv', 'imod', 'ipow',
                    'ilshift', 'irshift', 'iand', 'ior', 'ixor', 'missing']:
    method_name = '__%s__' % method_name
    new_method = types.MethodType(Frozen._illegal_modifier_method, None, Frozen)
    setattr(Frozen, method_name, new_method)

# Unary methods that return some non-primitive function of the object: legal,
# but special care required: they're normally fine to be delegated, but
# sometimes they could return the original object unchanged, in which case we
# need to wrap the return value.
#
# NOTE: This could be a problem for types where unary methods can return objects
# that somehow wrap or otherwise include references to the original object.
for method_name in ['pos', 'neg', 'abs', 'invert', 'round', 'floor', 'ceil',
                    'trunc']:
    method_name = '__%s__' % method_name
    new_method = types.MethodType(Frozen._get_unary_method(method_name), None,
                                  Frozen)
    setattr(Frozen, method_name, new_method)

# Reflected arithmetic operators: delegate, but allow to return NotImplemented
for method_name in [
    'radd', 'rsub', 'rmul', 'rfloordiv', 'rdiv', 'rtruediv', 'rmod', 'rdivmod',
    'rpow', 'rlshift', 'rrshift', 'rand', 'ror', 'rxor']:
    method_name = '__%s__' % method_name
    new_method = types.MethodType(Frozen._get_reflected_method(method_name),
                                  None, Frozen)
    setattr(Frozen, method_name, new_method)


# All other non-modifying methods: delegate directly to the underlying object.
# (Of course, we're trusting here that these functions have been defined
# sensibly, i.e., in a way that does not mutate the underlying object.)
# TODO: some of these (e.g., the pickling functions) will screw things up,
# because hasattr will return true but it would be false for the underlying
# object. Fix with properties?
for method_name in [
    'add', 'sub', 'mul', 'floordiv', 'div', 'truediv', 'mod', 'divmod', 'pow',
    'lshift', 'rshift', 'and', 'or', 'xor', 'dir', 'len', 'contains', 'enter',
    'exit', 'deepcopy', 'getinitargs', 'getnewargs', 'getstate', 'reduce',
    'reduce_ex', 'int', 'long', 'float', 'complex', 'oct', 'hex', 'index',
    'trunc']:
    method_name = '__%s__' % method_name
    new_method = types.MethodType(Frozen._get_legal_method(method_name), None,
                                  Frozen)
    setattr(Frozen, method_name, new_method)


def freeze(value, recursive=True):
    return Frozen(value, recursive)
