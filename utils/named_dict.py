# http://code.activestate.com/recipes/52308-the-simple-but-handy-collector-of-a-bunch-of-named/
# https://stackoverflow.com/questions/2827623/how-can-i-create-an-object-and-add-attributes-to-it
class NamedDict(dict):
    def __init__(self, d: dict = None, **kw):
        if d:
            kw.update(d)
        dict.__init__(self, kw)
        self.__dict__ = self


if __name__ == '__main__':
    def NamedDict_test():
        data = NamedDict({'d': 12}, ff=1, cc="1234 56")
        assert data.d == 12 and data.ff == 1 and data.cc == "1234 56"
        data = NamedDict({'d': 12}, ff=1, cc="1234 56")
        assert data.d == 12 and data.ff == 1 and data.cc == "1234 56"


    NamedDict_test()
