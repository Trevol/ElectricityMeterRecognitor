# http://code.activestate.com/recipes/52308-the-simple-but-handy-collector-of-a-bunch-of-named/
# https://stackoverflow.com/questions/2827623/how-can-i-create-an-object-and-add-attributes-to-it
class Bunch(dict):
    def __init__(self, **kw):
        dict.__init__(self, kw)
        self.__dict__ = self
