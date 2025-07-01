################################
#   Materialization utilities  #
################################

import gzip
import pickle
import base64
import hashlib
import importlib
import json
import numpy as np

# This file contains utilities to materialize the cache into a form that
# can be distributed

def do_compress_embeddings(cache, hash_keys):
    '''
    Find the embedding entries, and compress them
    '''

    rel_keys = [i for i in cache.keys() if 'embeddings' in json.loads(i)['stem']]

    for key in rel_keys:
        for i in range(len(cache[key])):
            for j in range(len(cache[key][i]['out'].data)):
                cache[key][i]['out'].data[j].embedding = np.array(cache[key][i]['out'].data[j].embedding).astype(np.float16)

        if hash_keys:
            cache[hashlib.md5(key.encode('utf-8')).hexdigest()] = cache[key]
            del cache[key]

def materialize_cache(cache               : dict,
                      delay_responses     : bool,
                      compress            : bool,
                      hash_keys           : bool,
                      file_name           : str | None,
                      used_keys           : list | None = None,
                      compress_embeddings : bool = False) -> str | None:
    '''
    This function materializes the cache into a pickle file for distribution. It accepts
    the following arguments:
      - cache : the cache to be saved
      - compress : whether the cache should be compressed
      - hash_keys : whether keys should be hashed; if yes, the dictionary keys will be hashed
                    and the pickle file will be smaller. On the other hand, the cache will then
                    become *final* - it will be impossible to add to it. If you choose to go
                    the hashed direction, it is recommended you first save the cache without
                    hashing the keys so that you can later add to it if you like
      - file_name : the file name to be used. If no filename is provided, a base64 encoded
                    string is returned
      - used_keys : a list of keys that should be included in the cache; any keys are that are
                    not included in this list are discarded
    '''

    # Ensure every entry in the cache is a list
    assert all([type(cache[i]) == list for i in cache])

    # Ensure every entry in those lists is a dict
    assert all([type(j) == dict for i in cache for j in cache[i]])

    # If we want used keys only, filter down the cache
    if used_keys:
        cache = {i:j for i, j in cache.items() if i in used_keys}

        # Remove any entries to pointers we've just removed
        for key in cache:
            cache[key] = [i for i in cache[key] if (list(i.keys()) != ['TARGET']) or (i['TARGET'] in cache)]

    # If we want to compress embeddings, do it
    if compress_embeddings:
        do_compress_embeddings(cache, hash_keys == False)

    # If we want to hash the keys, go ahead and do it
    if hash_keys:
        cache = {hashlib.md5(i.encode('utf-8')).hexdigest():j
                                        for i, j in cache.items()}
        
        # Make all pointers point to hashed values
        for key, vals in cache.items():
            for val in vals:
                if 'TARGET' in val:
                    val['TARGET'] = hashlib.md5(val['TARGET'].encode('utf-8')).hexdigest()

    # Pickle the dictionary
    pickled_data = pickle.dumps((delay_responses, cache))

    # Compress it if needed
    if compress:
        pickled_data = gzip.compress(pickled_data)
    
    # If the filename is provided, save it there; otherwise, return a b64 encoded
    # string
    if file_name:
        with open(file_name, 'wb') as f:
            f.write(pickled_data)
    
    else:
        return base64.b64encode(pickled_data).decode('utf-8')
    
def create_self_contained(cache               : dict,
                          delay_responses     : bool,
                          compress            : bool,
                          hash_keys           : bool,
                          file_name           : str | None,
                          used_keys           : list | None = None,
                          compress_embeddings : bool = False) -> str | None:
    '''
    Returns a self-contained .py file that includes the cache inside of it.

    See _materialized_cache for an explanation of the arguments
    '''

    # Create a placeholder for the output code
    out_code = []

    # Get the cache as a base 64 encoded string; do not include the file name to
    # ensure nothing is saved
    b64_cache = materialize_cache(cache, delay_responses, compress, hash_keys, None, used_keys)

    # Ensure only the OpenAI and AsyncOpenAI functions are exposed
    out_code.append("__all__ = ['OpenAI', 'AsyncOpenAI']")

    # Add the cache and the code to decompress it to the self-contained file
    out_code.append('import base64')
    out_code.append('import pickle')
    out_code.append(f'cache = base64.b64decode("{b64_cache}")')

    if compress:
        out_code.append('import gzip')
        out_code.append('cache = gzip.decompress(cache)')
        
    out_code.append('DELAY_RESPONSES, cache = pickle.loads(cache)')

    # Load the cached_client.py file
    with open(importlib.resources.files(__package__).joinpath('cached_client.py'), 'r') as f:
        out_code.extend(f.read().split('\n'))
    
    # Create the main entrypoints
    entrypoint_args = ( '                  api_key          ,'
                        'cache           = cache            ,'
                        'verbose         = False            ,'
                        'dev_mode        = False            ,'
                        'delay_responses = DELAY_RESPONSES  ,'
                        'temp_cache_file = None             ,'
                        'used_keys_file  = None             ')
    
    out_code.append(f'def OpenAI     (api_key : str | None = None) : return CachedClient({entrypoint_args}, is_async = False)')
    out_code.append(f'def AsyncOpenAI(api_key : str | None = None) : return CachedClient({entrypoint_args}, is_async = True)' )

    # Save the result
    with open(file_name, 'w') as f:
        f.write('\n'.join(out_code))