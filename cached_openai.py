# ==============================
# =   Cached OpenAPI Library   =
# =                            =
# =   Daniel Guetta            =
# =   daniel@guetta.com        =
# ==============================
#
# See readme file for usage instructions

import openai
import hashlib
import json
import pickle
import gzip


# Try and load the cache from disk; if no cache is found, start with an empty dictionary
try:
    with gzip.open('cache.pkl.gz', 'rb') as f:
        cache = pickle.load(f)
except:
    cache = {}

# Create a function to initialize the cached OpenAI client using
#       cached_openai.OpenAI(...)
def OpenAI(api_key : str, update_cache : bool = False):
    return CachedClient(api_key, update_cache=update_cache)

class CachedClient():
    '''
    This CachedClient object replicates the openai.OpenAI client object
    '''

    def __init__(self, api_key, stem=[], update_cache : bool = False):
        self._update_cache = update_cache
        self._api_key = api_key
        self._stem = stem

    def __getattr__(self, name):
        '''
        Any time we try to access an attribute of this class with a ., add the attribute
        to the stem, and return a new CachedClient object with that stem
        '''
        return CachedClient(api_key=self._api_key, stem=self._stem + [name], update_cache=self._update_cache)

    def __call__(self, *args, **kwargs):
        # Get the cache key, including the function name (stem) and arguments
        key = hashlib.md5(json.dumps({'step':self._stem, 'args':args, 'kwargs':kwargs}, sort_keys=True).encode('utf-8')).hexdigest()

        if key in cache:
            print('Saved result found; returning it')
            return cache[key]
        else:
            print('No saved result found; querying OpenAI')

        # Create a "real" openai.OpenAI client object
        rel_func = openai.OpenAI(api_key=self._api_key)

        # Go down the stem tree to find the relevant function
        for attr in self._stem:
            rel_func = getattr(rel_func, attr)

        # Call it
        out = rel_func(*args, **kwargs)

        # If we're update the cache, save the result there
        if self._update_cache:
            print('Saving result to the cache')
            cache[key] = out

        return out
    
    def save_cache(self):
        with gzip.open('cache.pkl.gz', 'wb') as f:
            pickle.dump(cache, f)