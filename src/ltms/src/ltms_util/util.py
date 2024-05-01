def iterflat(seq, levels=1):
    for itm in seq:
        if levels > 0 and isinstance(itm, type(seq)):
            yield from iterflat(itm, levels-1)
        else:
            yield itm

def iterwin(seq, winlen=1):
    slices = [seq[i::winlen] for i in range(winlen)]
    yield from zip(*slices)

def setdefaults(d: dict, *args, **kwds) -> None:
    """Set dictionary defaults."""
    if len(args) == 0:
        assert kwds, 'Missing arguments'
        defaults = kwds
    elif len(args) == 1:
        defaults, = args
        assert isinstance(defaults, dict), 'Single-argument form must be default dictionary'
        assert not kwds, 'Cannot supply keywords arguments with setdefault({...}) form'
    else:
        assert not kwds, 'Cannot supply keywords arguments with setdefault(key, val) form'
        assert len(args) % 2 == 0, 'Must have even number of arguments with setdefault(key, val) form'
        defaults = {key: val for key, val in iterwin(args, 2)}
    for key, val in defaults.items():
        d.setdefault(key, val)

