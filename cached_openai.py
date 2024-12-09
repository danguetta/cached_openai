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
# This is a global variable to ensure if various CachedClient objects are created, they
# all share the same cache
try:
    with gzip.open('cache.pkl.gz', 'rb') as f:
        cache = pickle.load(f)
except:
    cache = {}

# Create a function to initialize the cached OpenAI client using
#       cached_openai.OpenAI(...)
def OpenAI(api_key : str, update_cache : bool = False, strip_seed : bool = False):
    return CachedClient(api_key, update_cache=update_cache, strip_seed=strip_seed)

# Create an equivalent function for the asynchronous OpenAI client
def AsyncOpenAI(api_key : str, update_cache : bool = False, strip_seed : bool = False):
    return CachedClient(api_key, update_cache=update_cache, strip_seed=strip_seed, is_async=True)

class CachedClient():
    '''
    This CachedClient object replicates the openai.OpenAI client object
    '''

    def __init__(self, api_key, stem=[], update_cache : bool = False, strip_seed : bool = False, is_async : bool = False):
        self._api_key = api_key
        self._stem = stem
        self._update_cache = update_cache
        self._strip_seed = strip_seed
        self._is_async = is_async

    def __getattr__(self, name : str):
        '''
        Any time we try to access an attribute of this class with a ., add the attribute
        to the stem, and return a new CachedClient object with that stem
        '''
        return CachedClient(api_key      = self._api_key,
                            stem         = self._stem + [name],
                            update_cache = self._update_cache,
                            strip_seed   = self._strip_seed,
                            is_async     = self._is_async)

    def save_cache(self):
        '''
        Save the cache to disk
        '''
        with gzip.open('cache.pkl.gz', 'wb') as f:
            pickle.dump(cache, f)

    def get_cache_key(self, kwargs, strip_seed : bool = False):
        '''
        This function returns the cache key for the fuction described in self._stem called with
        parameters kwargs. If strip_seed is True, the seed parameter is removed from the kwargs
        '''

        if strip_seed:
            kwargs = {k:v for k,v in kwargs.items() if k != 'seed'}
        
        return hashlib.md5(json.dumps({'stem':self._stem, 'kwargs':kwargs}, sort_keys=True).encode('utf-8')).hexdigest()

    def read_from_cache(self, kwargs):
        '''
        This function will attempt to read the cached result for the function described in self._stem
        called with parameters kwargs.
        '''

        # Get the cache key, including the function name (stem) and arguments; do not remove the seed
        # even if self._strip_seed is true - that setting only refers to saving, not reading
        key = self.get_cache_key(kwargs)

        if key in cache:
            print('Saved result found')
            return cache[key]
        else:
            print('No saved result found')
            return None
    
    def write_to_cache(self, kwargs, out):
        '''
        This function will write the result of the function described in self._stem called with
        parameters kwargs to the cache. If self._strip_seed is True, it will be saved both *with*
        any seed parameter and without
        '''

        if self._update_cache:
            print('Saving result to the cache')

            cache[self.get_cache_key(kwargs)] = out

            if 'seed' in kwargs:
                cache[self.get_cache_key(kwargs, strip_seed=self._strip_seed)] = out

    def __call__(self, **kwargs):
        '''
        This function is called whenever an OpenAI function is called
        '''

        out = self.read_from_cache(kwargs)
        if out is not None:
            if self._is_async:
                async def async_func():
                    return out
                return async_func()
            else:
                return out
        
        # If we reached this point, we need to query OpenAI. Create a "real" openai.OpenAI client
        # object (sync or async as needed)
        if self._is_async:
            rel_func = openai.AsyncOpenAI(api_key=self._api_key)
        else:
            rel_func = openai.OpenAI(api_key=self._api_key)

        # Go down the stem tree to find the relevant function
        for attr in self._stem:
            rel_func = getattr(rel_func, attr)

        if self._is_async:
            async def async_func():
                out = await rel_func(**kwargs)
                self.write_to_cache(kwargs, out)
                return out
            
            return async_func()
        else:
            out = rel_func(**kwargs)
            self.write_to_cache(kwargs, out)
            return out