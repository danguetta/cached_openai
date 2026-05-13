########################################################################################################
# The library can be run in two modes, which affects its behavior.
#   - By default, it runs in prod mode
#   - If a CACHED_OPENAI_DEV_MODE environment variable is set (to any value), it runs in dev mode
# 
# The following points summarize the behavior of the library in both those settings
#   
#   - VERBOSITY
#     ---------
#         By default, the add-in runs in silent mode - no updates are printed. If the
#         CACHED_OPENAI_VERBOSE environment variable is set, various debugging messages will be
#         printed
#
#   - CACHE PRIORITY
#     --------------
#         When the library loads, it tries to do whatever it can to load a cache. It looks for
#         caches in the following order of priority:
#             * It looks for a file called CACHE_FILE_NAME in the current working directory
#             * It looks for a file called CACHE_FILE_NAME in the main package directory
#             * It checks if a cache_id is provided. If it is, it goes to CACHE_ID_SITE/cache_id,
#               gets a json blob from there, and looks at the "cache_location" element in that
#               dictionary. It then downloads the cache from there.
#             * If no cache_id is provided
#                  - In PROD MODE, an empty cache is used
#                  - In DEV MODE, an empty cache file is created, ready to receive entries
#
#   - SERVER CHOICE
#     -------------
#         In PROD MODE
#             * If the package name is of the form X_openai, the server will be X-openai.guetta.com.
#               if it exists
#             * If it doesn't exist, it goes to CACHE_ID_SITE/X, gets a json blob from there, and
#               looks at the "server" element in that dictionary.
#             * It then checks if a cache_id is provided. If it is, it goes to CACHE_ID_SITE/cache_id,
#               gets a json blob from there, and looks at the "server" element in that dictionary
#             * If no cache ID is provided, it uses the main OpenAI server
#
#         In DEV MODE
#             * The server will always be set as the main OpenAI server
#
#   - CALL BEHAVIOR
#     -------------
#         In PROD MODE
#              * First, we will look for the function call in the cache
#              * If it's not there, we will call the designed server, but NOT save the result
# 
#         In DEV MODE
#              * First, we will look for the function call in the cache
#              * If it's not there, we will call the designated server, and save the result to
#                the cache as well as the temporary cache file
#
########################################################################################################

import os
import datetime

from . import utils
from . import cached_client
from . import materialize_utils

###############
#  Constants  #
###############

# The name of the cache file
CACHE_FILE_NAME       = 'openai.cache'

# The name of the temporary cache and used keys files which will be written to in
# dev mode. These will be in the code's working directory
TEMP_CACHE_FILE_NAME  = 'openai_cache_temp.bin' 
USED_KEYS_FILE        = 'openai_cache_used.txt'

# The root website in which to look for cache ids
CACHE_ID_SITE         = 'https://xlkitlearn.guetta.com/ids'

####################
#  Prepare library #
####################
#

# Determine whether we want to print debugging information as we run the package;
# this is required if the 'CACHED_OPENAI_VERBOSE' variable is set in the
# environment
VERBOSE               = 'CACHED_OPENAI_VERBOSE' in os.environ

# In dev mode, all calls to the open AI API will be cached in the temporary cache
# file, and every time a request is made (either new or cached), we will track
# the fact the request has been made in a used keys file. The latter allows a
# "cleaning operation" that only keeps data for requests that have been made
# since this file was created.
#
# Dev mode is turned on if the 'CACHED_OPENAI_DEV_MODE' variable is set in the
# environment
DEV_MODE              = 'CACHED_OPENAI_DEV_MODE' in os.environ
if VERBOSE:
    print(f'DEV_MODE is set to {DEV_MODE}')



# When we recover values form the cache, we can either return the values instantly
# or delay for as long as it took to originally get a response from the OpenAI
# API. By default, no pausing will happen, unless a 'CACHED_OPENAI_DELAY_RESPONSES'
# variable is set in the environment 
DELAY_RESPONSES_NEW   = 'CACHED_OPENAI_DELAY_RESPONSES' in os.environ
if VERBOSE:
    print(f'DELAY_RESPONSES is set to {DELAY_RESPONSES_NEW}')

#################################################
#  Create the main entrypoints for the package  #
#################################################

def OpenAI(api_key : str | None = None, cache_id : str | None = None):
    delay_responses, cache = utils.get_cache(CACHE_FILE_NAME, TEMP_CACHE_FILE_NAME, DEV_MODE, DELAY_RESPONSES_NEW, VERBOSE, cache_id)

    return cached_client.CachedClient(                  api_key                   ,
                                      cache           = cache                     ,
                                      verbose         = VERBOSE                   ,
                                      dev_mode        = DEV_MODE                  ,
                                      is_async        = False                     ,
                                      delay_responses = delay_responses           ,
                                      temp_cache_file = TEMP_CACHE_FILE_NAME      ,
                                      used_keys_file  = USED_KEYS_FILE              )

def AsyncOpenAI(api_key : str | None = None, cache_id : str | None = None):
    delay_responses, cache = utils.get_cache(CACHE_FILE_NAME, TEMP_CACHE_FILE_NAME, DEV_MODE, DELAY_RESPONSES_NEW, VERBOSE, cache_id)

    return cached_client.CachedClient(                 api_key                    ,
                                      cache           = cache                     ,
                                      verbose         = VERBOSE                   ,
                                      dev_mode        = DEV_MODE                  ,
                                      is_async        = True                      ,
                                      delay_responses = delay_responses           ,
                                      temp_cache_file = TEMP_CACHE_FILE_NAME      ,
                                      used_keys_file  = USED_KEYS_FILE              )

##########################################
#  Create the materialization functions  #
##########################################

def materialize(self_contained      : bool,
                compress            : bool,
                hash_keys           : bool,
                used_keys_only      : bool,
                compress_embeddings : bool):
    '''
    This function materilizes the cache and prepares it for distribution. It accepts
    the following options
      - self_contained : if True, the function will create a file called
        cached_openai_{date}.py in the current working directory which will contain
        the materialized cache. This file will be a self-contained .py file containing
        a serialized verison of the cache. Just import that package, and you're good
        to go with the full cache.

        If False, the function will create a file called named openai_{date}.cache
        containing the cache only. The file can be distributed in three ways
          - Rename it to openai.cache, and ask your target audience to place the file
            in their working directory. They can then install cached_openai through
            pypi, and when they load the package, it will find the file in the working
            directory and load it
          - Upload it to some URL. Then, get your audience to install cached_openai
            from pypi. When they load the package, it will prompt them for the URL,
            from which the file will autoomatically be downloaded
          - Fork cached_openai, rename the file to openai.cache, and include it in
            the src directory of the package. Publish it. When your target audience
            downloads the package, the cache will be read from this file.

      - compress: True if the cache should be compressed before saving or false
        otherwise. Note that this means the package will take longer to run every time
        it is loaded 
      
      - hash_keys : whether keys should be hashed; if True, the dictionary keys will be hashed
        and the pickle file will be smaller. On the other hand, the cache will then become
        *final* - it will be impossible to add to it. If you choose to go the hashed direction,
        it is recommended you first save the cache without hashing the keys so that you can
        later add to it if you like

      - used_keys_only : if True, the cache will only contain the keys that have been
        created or access since the last time the openai_cache_used.txt file was created.

        This is useful if you fill the cache with a whole bunch of data while you test
        your code, and then want to create a "clean" version to distribute. Just delete the
        openai_cache_used.txt file, run you clean code, and then run this with used_keys_only
        = True.
      - compress_embeddings : if True, embeddings will be compressed to float16's, and the
        keys of embedding entries will be hashed
    '''

    current_date = datetime.datetime.now().strftime('%Y-%m-%d')

    if used_keys_only:
        with open(USED_KEYS_FILE, 'r') as f:
            used_keys = f.read().split('\n')
    else:
        used_keys = None

    if self_contained:
        materialize_utils.create_self_contained(cache               = cache,
                                                delay_responses     = delay_responses,
                                                compress            = compress,
                                                hash_keys           = hash_keys,
                                                file_name           = f'cached_openai_{current_date}.py',
                                                used_keys           = used_keys,
                                                compress_embeddings = compress_embeddings)
    else:
        materialize_utils.materialize_cache(cache               = cache,
                                            delay_responses     = delay_responses,
                                            compress            = compress,
                                            hash_keys           = hash_keys,
                                            file_name           = CACHE_FILE_NAME.split('.')[0]
                                                                        + f'_{current_date}.'
                                                                            + CACHE_FILE_NAME.split('.')[-1],
                                            used_keys           = used_keys,
                                            compress_embeddings = compress_embeddings)