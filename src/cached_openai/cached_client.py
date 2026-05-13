import openai
import hashlib
import json
import os
import base64
import pathlib
import time
import asyncio
import pickle
import copy
import requests
import io
import struct
import inspect
import numpy as np

# There are some keywords that - when provided to an OpenAI function - do not change
# the result; we should ignore these completely when caching results
IRRELEVANT_KWARGS     = ['timeout', 'delay']

class CachedClient():
    '''
    This CachedClient object replicates the openai.OpenAI client object, but allows the loading
    and saving of results to or from cache every time a request is made.

    It can be created in two circumstances:
      - When it is created by the user, it will be created with stem = []
      - When the user accesses a method of this class, a new class is recursively created with
        the stem extended by the attribute accessed. For example, if the user calls
            client.chat.completion.create
        the last CachedClient instance will have stem = ['chat', 'completion', 'create']. This
        final instance can then be called, which will called the corresponding function in the
        original OpenAI library
    '''

    def __init__(self,
                 api_key             : str | None     ,
                 cache               : dict           ,
                 verbose             : bool           ,
                 dev_mode            : bool           ,
                 is_async            : bool           ,
                 delay_responses     : bool           ,
                 temp_cache_file     : str            , 
                 used_keys_file      : str            ,
                 stem                : list[str] = [] ,
                 last_entry_returned : dict      = {}  ):
        
        # Store variables
        self._api_key         = api_key
        self._cache           = cache
        self._verbose         = verbose
        self._dev_mode        = dev_mode
        self._is_async        = is_async
        self._delay_responses = delay_responses
        self._temp_cache_file = temp_cache_file
        self._used_keys_file  = used_keys_file
        self._stem            = stem

        # If we were not given an API key, check whether one is available in an openai_key.txt file
        if self._api_key is None:
            try:
                with open('openai_key.txt', 'r') as f:
                    self._api_key = f.read().strip()
                
                print('Read API key from openai_key.txt')
                print()
            except:
                pass

        if __package__ != 'cached_openai':
            required_prefix = __package__.split('_')[0]

            # If we're calling this directly (i.e., if the stem is []), ensure the key is an appropriate
            # key and print a warning in re: the intermediate server
            if (self._api_key is not None) and (len(self._stem) == 0):        
                if not self._api_key.startswith(required_prefix):
                    raise BaseException('\nYou are trying to use this library with an OpenAI key. You should not provide\n'
                                        'your OpenAI key to any library other than the official openai library. This\n'
                                        'library is only meant to be used with the API key provided by your instructor,\n'
                                       f'which will always start wtih the characters "{required_prefix}-".')
                else:
                    # The API key starts with the correct text; transform it to the actual key
                    self._api_key = 'sk' + self._api_key[len(required_prefix):]

                print('WARNING : You will be making requests via the class server. These requests may be\n'
                      '          logged. Do *NOT* make any requests with confidential or sensitive data.')


        # In some cases, we have multiple results for a single set of keys; this is so that
        # we can simulate the "real" OpenAI API that would return different results every time
        # it is run. Initialize a dictionary to store how many responses we've returned for a
        # given key, so that we know the next one we should return next time it is called
        self._last_entry_returned = last_entry_returned

    def check_remaining_credits(self) -> dict:
        '''
        Make a request from the proxy server for remaining credits
        '''
        import requests
        out = requests.get(f'https://{__package__.replace("_","-")}.guetta.com/key/info', headers={'x-litellm-api-key' : f'Bearer {self._api_key}'}).json()
        out = out['info']
        
        return {'max_budget'       : out['max_budget'],
                'spent_so_far'     : out['spend'],
                'remaining_budget' : out['max_budget'] - out['spend']}

    def __getattr__(self, name : str):
        '''
        This function is called whenever an instance of this class is accessed with a .;
        for example, client.chat.

        When this happens, we add the attribute being accessed to self._stem, and return
        a new CachedClient instance with that new stem.
        '''

        return CachedClient(api_key             = self._api_key,
                            cache               = self._cache,
                            verbose             = self._verbose,
                            dev_mode            = self._dev_mode,
                            is_async            = self._is_async,
                            delay_responses     = self._delay_responses,
                            temp_cache_file     = self._temp_cache_file,
                            used_keys_file      = self._used_keys_file,
                            stem                = self._stem + [name],
                            last_entry_returned = self._last_entry_returned    )

    def get_cache_key(self, kwargs, hash_key : bool, strip_seed : bool = False):
        '''
        This function returns the cache key for the fuction described in self._stem called with
        parameters kwargs. If hash_key is True, the JSon key will be hashed, otherwise it will
        be returned raw
        
        The following modifications are made:
          - If strip_seed is True, the seed parameter is removed from the kwargs
          - Any of the arguments in IRRELEVANT_KWARGS are in kwargs, they are removed
          - If 'with_raw_response' is in the stem, it is stripped from it; when this is is included
            in an OpenAI API call, it is because the developer wants to get the raw response, which
            includes the number of token's left in the user's quota. It makes no sense to store this
            in the cache as it will be different every time
        '''

        if strip_seed:
            kwargs = {k:v for k,v in kwargs.items() if k != 'seed'}
        
        this_stem = self._stem
        this_stem = [i for i in this_stem if i != 'with_raw_response']

        # Remove any irrelevant kwargs
        kwargs = {k:v for k,v in kwargs.items() if k not in IRRELEVANT_KWARGS}

        key = json.dumps({'stem':this_stem, 'kwargs':kwargs}, sort_keys=True)
        if hash_key:
            return hashlib.md5(key.encode('utf-8')).hexdigest()
        else:
            return key

    def read_from_cache(self, kwargs):
        '''
        This function will attempt to read the cached result for the function described in self._stem
        called with parameters kwargs.

        It will return a dictionary with two entries:
          - out : the entry in question
          - run_time : the time the function took to run when it was initially added to the cache
        If the entry is not found int he cache, None is returned
        '''

        # Remove any irrelevant kwargs
        kwargs = {k:v for k,v in kwargs.items() if k not in IRRELEVANT_KWARGS}

        # Try and the find the value in the cache; first, look for the raw JSon, and if it's not found
        # look for the hashed key. Do NOT strip the seed - if the user intentionally added a seed argument,
        # we want THAT entry specifically
        key = self.get_cache_key(kwargs, hash_key = False)
        if key not in self._cache:
            key = self.get_cache_key(kwargs, hash_key = True)
        
        # Check whether we have a result
        if key in self._cache:
            if self._verbose:
                print('Found a saved result in the cache')
            
            # Retrieve the entry from the cache
            cache_entry = self._cache[key]

            # Check whether this cache_entry is of one of two specific types:
            #   - A pointer (a dictionary with a single entry with the key 'TARGET'), pointing to
            #     another entry in the cache. If we have such an entry, we need to follow it
            #   - A list, in which case there are many possible results for this key, and we need
            #     to return the next one

            while (type(cache_entry) == list) or ('TARGET' in cache_entry):
                if type(cache_entry) == list:
                    # Find the next entry to retrieve from the list, wrapping back to the front of
                    # the list if we reach the end
                    self._last_entry_returned[key] = (self._last_entry_returned.get(key, -1) + 1) % len(cache_entry)
                    cache_entry = cache_entry[self._last_entry_returned[key]]
                elif 'TARGET' in cache_entry:
                    # Log the fact we used this key, and then follow the pointer
                    if self._dev_mode:
                        with open(self._used_keys_file, 'a') as f : f.write(key + '\n')
                    
                    key = cache_entry['TARGET']
                    cache_entry = self._cache[key]
            
            # Record the fact we've used the key
            if self._dev_mode:
                with open(self._used_keys_file, 'a') as f: f.write(key + '\n')
            
            # Retrieve the output that was saved from OpenAI
            out = cache_entry['out']

            # If the cache_entry contains a 'saved_images' entry, handle the images returned by the
            # API
            if 'saved_images' in cache_entry:
                # Make sure we don't mutate the original object in the cache
                out = copy.deepcopy(out)

                # Get the saved images that were downloaded the cache
                saved_images = cache_entry['saved_images']

                # out.data is the entry in the OpenAI object that contains the image URLs. saved_images
                # contains the actual image data. We want to save those images as a file, and replace
                # the URL in out.data with the URL of the new file
                for im, saved_im in zip(out.data, saved_images):
                    if saved_im is not None:
                        # Create a file name for this image based on the hash of the URL
                        file_name = hashlib.md5((key + im.url).encode('utf-8')).hexdigest() + '.png'

                        # Check whether the images folder exists; if not, create it
                        if not os.path.exists('images'):
                            os.mkdir('images')

                        # Save the image there
                        with open(f'images/{file_name}', 'wb') as f:
                            f.write(saved_im)

                        # Alter the URL in the output object
                        im.url = pathlib.Path(f'images/{file_name}').resolve().as_uri()

            # If the cache_entry contains an 'audio_file' entry, deal with the audio file
            if 'audio_file' in cache_entry:
                # Create a class that will allow us to use a iter_bytes method and a stream_to_file
                # method
                class Stream:
                    def __init__(self, bytes):
                        self.byte_stream = io.BytesIO(base64.b64decode(bytes.encode('utf-8')))
                    
                    def iter_bytes(self):
                        while True:
                            chunk = self.byte_stream(1024)
                            if not chunk:
                                break
                            yield chunk
                    
                    def stream_to_file(self, file_name):
                        with open(file_name, 'wb') as f:
                            f.write(self.byte_stream.getvalue())


                if cache_entry['audio_file'][0] == 'old':
                    return {'out'      : Stream(cache_entry['audio_file'][1]),
                            'run_time' : cache_entry['run_time']               }
                
                elif cache_entry['audio_file'][0] == 'new':
                    # Create a class that we can use as a context manager to return the Stream
                    # object
                    class AudioFile:
                        def __enter__(self):
                            return Stream(cache_entry['audio_file'][1])
                            
                        def __exit__(self, exc_type, exc_value, traceback):
                            pass

                        def __call__(self):
                            return self
                    
                    return {'out'      : AudioFile(),
                            'run_time' : cache_entry['run_time']}

            # If we have an embedding that was saved as a lower-accuracy numpy array, reconvert
            # it to a list
            if type(out) == openai.types.create_embedding_response.CreateEmbeddingResponse:
                for i in out.data:
                    if type(i.embedding) == np.ndarray:
                        i.embedding = list(i.embedding)

            # If we reached this point, we don't have an audio file - return
            return {'out'      : out,
                    'run_time' : cache_entry['run_time']}
        else:
            if self._verbose:
                print('No saved result found')

            return None

    def modify_cache(self, key, value):
        '''
        This function writes a specific key and value to the cache and the temporary cache
        file, and records the fact the key has been used
        '''

        # Record in the cache
        self._cache[key] = value
        
        # Save the value to the temporary cache file 
        with open(self._temp_cache_file, 'ab') as f:
            # Get the entry
            entry = pickle.dumps([key, value])

            # Write its length to the file
            f.write(struct.pack('I', len(entry)))

            # Then, write the entry
            f.write(entry)

        # Record the fact the key has been used
        with open(self._used_keys_file, 'a') as f:
            f.write(key + '\n')

    def write_to_cache(self, kwargs, out, run_time):
        '''
        If we are in dev mode, this function will write the result of the function
        described in self._stem called with parameters kwargs to the cache.

        It also adds the result ot the temporary cache file
        '''

        if self._dev_mode:
            if self._verbose:
                print('Saving result to the cache')

            # If we asked for the raw response from the OpenAI API, get the parsed response - we
            # do NOT want to save the raw response to the cache, because it contains things like
            # the number of tokens remaining, which won't be relevant/valid when the value is
            # pulled from the ache
            if 'with_raw_response' in self._stem:
                out = out.parse()
            
            # Prepare the output object
            out_obj = {'out':out, 'time_saved':time.time(), 'run_time':run_time}

            # Check whether this is a request throught the image API - if so, we need to check
            # whether URLs were returned; if they were, we should save them
            if 'images' in self._stem:
                saved_images = []
                for im in out.data:
                    if im.url:
                        saved_images.append(requests.get(im.url).content)
                    else:
                        saved_images.append(None)
                out_obj['saved_images'] = saved_images

            # Check whether this is a request throught he audio API - if so, we need to download
            # the resulting file, and save it. Unfortunately, there are two ways this API might
            # be called - the legacy way (client.audio.speech.create) and the new way (client.
            # audio.speech.with_streaming_response.create). They each require different ways to
            # download the file
            if type(out) == openai._legacy_response.HttpxBinaryResponseContent:
                # The user used the legacy format; get the file in base64 format
                audio_data = io.BytesIO()
                for chunk in out.iter_bytes():
                    audio_data.write(chunk)
                audio_data.seek(0)
                audio_data = audio_data.read()
                audio_data = base64.b64encode(audio_data).decode('utf-8')

                out_obj['audio_file'] = ('old', audio_data)
            
            if type(out) == openai._response.ResponseContextManager:
                # The user used the new format; get the file in base64 format
                audio_data = io.BytesIO()
                with out as _out:
                    for chunk in _out.iter_bytes():
                        audio_data.write(chunk)
                audio_data.seek(0)
                audio_data = audio_data.read()
                audio_data = base64.b64encode(audio_data).decode('utf-8')

                out_obj['audio_file'] = ('new', audio_data)

            # First, save the entry as provided
            seeded_key = self.get_cache_key(kwargs, hash_key=False)

            if 'seed' in kwargs:
                # If this call includes a seed, we just want to overwrite whatever already exists in
                # the cache at that position
                self.modify_cache(seeded_key, [out_obj])

                # Now, strip the seed, and look at the corresponding entry - if a pointer to this
                # seeded entry doesn't yet exist there, add it
                stripped_key = self.get_cache_key(kwargs, strip_seed=True, hash_key=False)

                current_pointers = [i['TARGET'] for i in self._cache.get(stripped_key, []) if 'TARGET' in i]

                if seeded_key not in current_pointers:
                    self.modify_cache(stripped_key, self._cache.get(stripped_key, []) + [{'TARGET':seeded_key}])
            else:
                # This isn't a seeded request. If we already have an entry there, append this request
                # to the list
                self.modify_cache(seeded_key, self._cache.get(seeded_key, []) + [out_obj])
                
    def __call__(self, **kwargs):
        '''
        This function is called whenever an OpenAI function is called
        '''

        # Try and read the value from the cache
        out = self.read_from_cache(kwargs)

        if out is not None:
            # We were able to pull a value from the cache; return either the value, or an async
            # funcion that returns it. Pause if needed.

            if self._is_async:
                async def async_func():
                    if self._delay_responses or ('delay' in kwargs and kwargs['delay']):
                        await asyncio.sleep(out['run_time'])
                    return out['out']
                return async_func()
            
            else:
                if self._delay_responses or ('delay' in kwargs and kwargs['delay']):
                    time.sleep(out['run_time'])

                if 'stream' in kwargs and kwargs['stream']:
                    def make_generator():
                        for i in out['out']:
                            time.sleep(i[0])
                            yield i[1]
                    
                    return make_generator()
                else:
                    return out['out']
        
        # If we reached this point, we need to query OpenAI. Make sure we have an OpenAI key
        if self._api_key is None:
            raise ValueError('Your request is not available in the cache, and you did not provide '
                             "an API key, so I can't query OpenAI for you.")
        
        # Create a "real" openai.OpenAI client object (sync or async as needed)
        if __package__ == 'cached_openai':
            if self._is_async:
                rel_func = openai.AsyncOpenAI(api_key=self._api_key)
            else:
                rel_func = openai.OpenAI(api_key=self._api_key)
        else:
            if self._is_async:
                rel_func = openai.AsyncOpenAI(api_key=self._api_key, base_url=f'https://{__package__.replace("_","-")}.guetta.com')
            else:
                rel_func = openai.OpenAI(api_key=self._api_key, base_url=f'https://{__package__.replace("_","-")}.guetta.com')
                
        # Go down the stem tree to find the relevant function
        for attr in self._stem:
            rel_func = getattr(rel_func, attr)

        # If the function was called with a seed but the OpenAI function does not accept one,
        # strip it before calling
        kwargs_copy = {i:j for i, j in kwargs.items()}
        if 'seed' not in inspect.signature(rel_func).parameters:
            if 'seed' in kwargs_copy:
                if self._verbose:
                    print('Detected a seed parameter in an OpenAPI call that does not accept a seed. '
                          "I'll strip the parameter from the call before sending it to OpenAI, but "
                          "save it in the cached response. See the user manual (section 'repeated "
                          "requests') for details" )
                kwargs_copy = {i:j for i, j in kwargs.items() if i != 'seed'}
        
        # Remove a delay parameter if it exists
        if 'delay' in kwargs_copy:
            del kwargs_copy['delay']

        # Call it, write the result to the cache, and return either the value or the co-routine
        # if we are in async mode
        if self._is_async:
            async def async_func():
                start_time = time.time()
                out = await rel_func(**kwargs_copy)
                self.write_to_cache(kwargs, out, time.time() - start_time)
                return out
            
            return async_func()
        
        else:
            start_time = time.time()

            if ('stream' in kwargs) and kwargs['stream']:
                def make_generator():
                    out_ = rel_func(**kwargs_copy)

                    out = []
                    last_time = time.time()
                    for i in out_:
                        out.append((time.time() - last_time, i))
                        last_time = time.time()
                        yield i
                    
                    self.write_to_cache(kwargs, out, time.time() - start_time)

                return make_generator()

            else:
                out = rel_func(**kwargs_copy)

                self.write_to_cache(kwargs, out, time.time() - start_time)

                return out