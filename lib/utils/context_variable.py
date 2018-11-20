class ContextVariable(object):

    def __init__(self):
        self._cxt = {}
        self._cxt_prefix = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self._cxt_prefix = None

    def __call__(self, prefix=None):
        self._cxt_prefix = prefix
        return self

    def resolve(self, name, prefix):
        if self._cxt_prefix is not None:
            name = '{}{}'.format(self._cxt_prefix, name)
        elif prefix is not None:
            name = '{}{}'.format(prefix, name)
        return name

    def register(self, name, value, prefix=None):
        name = self.resolve(name, prefix)
        self._cxt[name] = value
        return value

    def access(self, name, prefix=None):
        name = self.resolve(name, prefix)
        return self._cxt[name]

    def check(self, keys=[], prefix=None):
        for key in keys:
            key = self.resolve(key, prefix)
            assert key in self._cxt

    def dump(self):
        return self._cxt
