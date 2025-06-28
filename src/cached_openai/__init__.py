# Make the main entrypoints into the package available at the top level of the package
from .main import OpenAI, AsyncOpenAI, DEV_MODE

# Make the materialization functions available at the top level of the package, if
# we're in dev mode
if DEV_MODE:
    from .main import materialize

# Remove DEV_MODE from the namespace
del DEV_MODE

"""
class ResponsePointer():
    '''
    This class is used to store duplicated cache entries without wasting space; if two entries
    have the same value, one will simply contain a ResponsePointer pointing to the other
    '''

    def __init__(self, target):
        self.target = target
"""