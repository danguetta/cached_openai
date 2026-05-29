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
import httpx
import io
import struct
import inspect
import numpy as np
import PIL.Image

import pydantic._internal._model_construction
import warnings
import importlib

# There are some keywords that - when provided to an OpenAI function - do not change
# the result; we should ignore these completely when caching results
IRRELEVANT_KWARGS     = ['timeout', 'delay', 'overwrite_cache']

# Some parameters are only used internally by this cache library and should not be
# passed to OpenAI
NON_OPENAI_PARAMS     = ['delay', 'overwrite_cache']

# If OpenAI sends extraneous headers to OpenRouter, then this library can't be use in pyodide
# in Excel online because CORS will block the sending of these headers to OpenRouter. Monkey
# patch openai not to send them; make the patch idempotent
import openai._base_client as bc
if not hasattr(bc.BaseClient, "_original_build_headers"):
    bc.BaseClient._original_build_headers = bc.BaseClient._build_headers            # type: ignore
 
def _cors_safe_build_headers(self, options, *, retries_taken=0):
    headers = self._original_build_headers(options, retries_taken=retries_taken)

    for h in [
        "x-stainless-read-timeout",
        "x-stainless-retry-count",
        "x-stainless-lang",
        "x-stainless-package-version",
        "x-stainless-os",
        "x-stainless-arch",
        "x-stainless-runtime",
        "x-stainless-runtime-version",
        "x-stainless-async",
    ]:
        headers.pop(h, None)
        headers.pop(h.title(), None)

    return headers

bc.BaseClient._build_headers = _cors_safe_build_headers

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
                 base_url            : str | None     ,
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
        self._base_url        = base_url
        self._cache           = cache
        self._verbose         = verbose
        self._dev_mode        = dev_mode
        self._is_async        = is_async
        self._delay_responses = delay_responses
        self._temp_cache_file = temp_cache_file
        self._used_keys_file  = used_keys_file
        self._stem            = stem
        
        # In some cases, we have multiple results for a single set of keys; this is so that
        # we can simulate the "real" OpenAI API that would return different results every time
        # it is run. Initialize a dictionary to store how many responses we've returned for a
        # given key, so that we know the next one we should return next time it is called
        self._last_entry_returned = last_entry_returned

    def __getattr__(self, name : str):
        '''
        This function is called whenever an instance of this class is accessed with a .;
        for example, client.chat.

        When this happens, we add the attribute being accessed to self._stem, and return
        a new CachedClient instance with that new stem.
        '''

        return CachedClient(api_key             = self._api_key,
                            base_url            = self._base_url,
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

        pydantic._internal._model_construction.ModelMetaclass
        def json_parse_fallback(obj):
            if isinstance(obj, pydantic._internal._model_construction.ModelMetaclass):
                return obj.model_json_schema()
            raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

        key = json.dumps({'stem':this_stem, 'kwargs':kwargs}, sort_keys=True, default=json_parse_fallback)
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
        unhashed_key = None
        if key not in self._cache:
            unhashed_key = key
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
                        if unhashed_key is not None: 
                            with open('dehash_' + self._used_keys_file, 'a') as f : f.write(key + ':' + unhashed_key + '\n')
                    
                    key = cache_entry['TARGET']
                    cache_entry = self._cache[key]
            
            # Record the fact we've used the key
            if self._dev_mode:
                with open(self._used_keys_file, 'a') as f: f.write(key + '\n')
                if unhashed_key is not None:
                    with open('dehash_' + self._used_keys_file, 'a') as f : f.write(key + ':' + unhashed_key + '\n')
            
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

            # If we saved a structured output response, unpack it
            if (type(out) == dict) and ('parsed_chat_completion_data' in out):
                out = openai.lib._parsing._completions.parse_chat_completion(
                    response_format = kwargs.get('response_format', openai.NOT_GIVEN),
                    input_tools = kwargs.get('tools', []),
                    chat_completion=openai.types.chat.ChatCompletion.model_validate(out['parsed_chat_completion_data'])
                )

            # If we reached this point, we don't have a "special" output - return
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

            # If we have a structured output response, it can't be pickled natively - we need to
            # serialize it in a very specific way; for some reason, model_dump triggers a warning
            # to suppress that - it still works for our purposes
            if isinstance(out, openai.types.chat.parsed_chat_completion.ParsedChatCompletion):
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', category=UserWarning, message=r'Pydantic serializer warnings:.*')
                    out = {'parsed_chat_completion_data' : out.model_dump(mode='python')}
                    
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
                # This isn't a seeded request. Each entry should have at most one non-seeded request;
                # replace it
                if seeded_key not in self._cache:
                    out_cache = [out_obj]
                else:
                    cur_cache = self._cache[seeded_key]
                    non_pointer_entries = [i_n for i_n, i in enumerate(cur_cache) if 'TARGET' not in i]

                    if len(non_pointer_entries) == 0:
                        out_cache = cur_cache + [out_obj]
                    elif len(non_pointer_entries) == 1:
                        out_cache = list(cur_cache)
                        out_cache[non_pointer_entries[0]] = out_obj
                    else:
                        raise f'Multiple non-pointer entries found. seeded_key was {seeded_key}'
                
                self.modify_cache(seeded_key, out_cache)
                
    def __call__(self, **kwargs):
        '''
        This function is called whenever an OpenAI function is called
        '''

        # Try and read the value from the cache
        if kwargs.get('overwrite_cache') == True:
            out = None
        else:
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
                             "an API key, so I can't run your request.")
        
        # gemini-embedding-2-preview is an important model, but unfortunately it doesn't
        # return in a format that is compatible with the OpenAI SDK. So we need to create a
        # shim to make it work
        if (self._stem == ['embeddings', 'create']) and (kwargs.get('model', '') in ['google/gemini-embedding-2-preview', 'gemini-embedding-2-preview']):
            def make_gemini_embedding_payload(**kwargs):
                return {'url'     : "https://openrouter.ai/api/v1/embeddings",
                        'headers' : {
                                        "Authorization": f"Bearer {self._api_key}",
                                        "Content-Type": "application/json"
                                    },
                        'data'    : json.dumps({
                                    "model": "google/gemini-embedding-2-preview",
                                    "input": kwargs['input']
                                })}

            def process_gemini_embedding_response(resp):
                return openai.types.create_embedding_response.CreateEmbeddingResponse(
                        data     = [openai.types.embedding.Embedding(object    = i['object'],
                                                                     embedding = i['embedding'],
                                                                     index     = i['index']) for i in resp['data']],
                        model    = resp['model'],
                        object   = resp['object'],
                        usage    = openai.types.create_embedding_response.Usage(prompt_tokens = resp['usage']['prompt_tokens'],
                                                                                total_tokens  = resp['usage']['total_tokens'],
                                                                                cost          = resp['usage']['cost'] ),
                        provider = resp['provider'],
                        id       = resp['id']
                )

            if self._is_async:
                async def rel_func(**kwargs):
                    async with httpx.AsyncClient() as client:
                        resp = (await client.post(**make_gemini_embedding_payload(**kwargs))).json()
                    return process_gemini_embedding_response(resp)
            else:
                def rel_func(**kwargs):
                    resp = requests.post(**make_gemini_embedding_payload(**kwargs)).json()
                    
                    return process_gemini_embedding_response(resp)
                    
        else:
            # Create a "real" openai.OpenAI client object (sync or async as needed)
            if self._is_async:
                rel_func = openai.AsyncOpenAI(api_key=self._api_key, base_url=self._base_url)
            else:
                rel_func = openai.OpenAI(api_key=self._api_key, base_url=self._base_url)
            
            # Go down the stem tree to find the relevant function
            for attr in self._stem:
                rel_func = getattr(rel_func, attr)

        # Make a copy of the kwargs
        kwargs_copy = {i:j for i, j in kwargs.items()}

        # Remove non-open-AI parmeters if they exist
        kwargs_copy = {i:j for i, j in kwargs_copy.items() if i not in NON_OPENAI_PARAMS}

        # If the function was called with a seed but the OpenAI function does not accept one,
        # strip it before calling
        if 'seed' not in inspect.signature(rel_func).parameters:
            if 'seed' in kwargs_copy:
                if self._verbose:
                    print('Detected a seed parameter in an OpenAPI call that does not accept a seed. '
                          "I'll strip the parameter from the call before sending it to OpenAI, but "
                          "save it in the cached response. See the user manual (section 'repeated "
                          "requests') for details" )
                kwargs_copy = {i:j for i, j in kwargs_copy.items() if i != 'seed'}

        # Create a function that converts any base64 encoded images to PIL images
        def decode_image_in_response(resp):
            if type(resp) == openai.types.chat.chat_completion.ChatCompletion:
                for choice in resp.choices:
                    if hasattr(choice.message, 'images'):
                        for image_n in range(len(choice.message.images)):
                            image = choice.message.images[image_n]
                            if image.get('type') == 'image_url':
                                image_url = image.get('image_url',{}).get('url') or ''
                                if image_url.startswith('data:image/png;base64,'):
                                    choice.message.images[image_n] = PIL.Image.open(io.BytesIO(base64.b64decode(image_url.split(",", 1)[1])))
            return resp
        
        # Call it, write the result to the cache, and return either the value or the co-routine
        # if we are in async mode
        if self._is_async:
            async def async_func():
                start_time = time.time()
                out = await rel_func(**kwargs_copy)
                out = decode_image_in_response(out)
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
                out = decode_image_in_response(rel_func(**kwargs_copy))

                self.write_to_cache(kwargs, out, time.time() - start_time)

                return out