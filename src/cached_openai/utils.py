import requests
import gzip
import tqdm
import os
import importlib.resources
import pickle
import struct
from tqdm import tqdm

def download_cache(cache_url : str, target_file : str) -> None:
    '''
    This function takes the URL of a cache file, downloads it (with a progress bar) and saves
    it to the target_file. If the URL extension ends with .gz, the file is unzipped as it
    downloads
    '''

    # If the URL is incomplete, prepend https://
    if not (cache_url.startswith('http://') or cache_url.startswith('https://')):
        cache_url = 'https://' + cache_url

    # Check whether the file is a .gz file
    is_gz = cache_url.endswith('.gz')

    # Start the request
    with requests.get(cache_url, stream=True) as response:
        # Raise an error if the request was unsuccessful
        response.raise_for_status()

        # Get the total size and prepare the progress bar
        total_size = int(response.headers.get('content-length', 0))
        progress_bar = tqdm.tqdm(total=total_size, unit='B', unit_scale=True, desc='Downloading cache file')

        if is_gz:
            # Monkey patch the read method
            original_read = response.raw.read
            def new_read(chunk_size=-1):
                data = original_read(chunk_size)
                if data:
                    progress_bar.update(len(data))
                return data
            response.raw.read = new_read

            with gzip.GzipFile(fileobj=response.raw, mode='rb') as gz_in, open(target_file, 'wb') as f:
                while True:
                    data = gz_in.read(8192)
                    if not data:
                        break
                    f.write(data)

        else:
            with open(target_file, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        progress_bar.update(len(chunk))
        
        progress_bar.close()

class TqdmFileReader:
    '''
    This file reader class wraps a file object and provides a tqdm progress bar
    '''

    def __init__(self, file_path):
        self.f = open(file_path, "rb")
        self.total = os.path.getsize(file_path)
        self.pbar = tqdm(total=self.total, unit='B', unit_scale=True)
    
    def read(self, size=-1):
        chunk = self.f.read(size)
        self.pbar.update(len(chunk))
        return chunk

    def readline(self, *args, **kwargs):
        line = self.f.readline(*args, **kwargs)
        self.pbar.update(len(line))
        return line

    def __getattr__(self, attr):
        return getattr(self.f, attr)

    def close(self):
        self.f.close()
        self.pbar.update(self.total - self.pbar.n)
        self.pbar.close()

def load_cache_file(cache_loc : str) -> tuple[bool, dict]:
    '''
    This function accept a bytes string containing a pickled object; if first checks
    whether it can be loaded uncompressed, and if not, it tries to load it as a
    compressed file
    
    It returns the object from the pickle
    '''

    try:
        reader = TqdmFileReader(cache_loc)
        obj = pickle.load(reader)
        reader.close()

        return obj
    except:
        try:
            with open(cache_loc, 'rb') as f : data = f.read()
            return pickle.loads(gzip.decompress(data))
        except:
            raise ValueError('Invalid cache file provided')

def get_cache(cache_file_name      : str          ,
              temp_cache_file_name : str          ,
              dev_mode             : bool         ,
              delay_responses_new  : bool         ,
              verbose              : bool = False  ) -> tuple[bool, dict]:
    '''
    This function attempts to load the cache from disk, in the following order of priority
      - First, we look in the working directory
      - Then, we look in the directory containing the package
      - Finally, two options
          * If we are in dev mode, we create a blank cache
          * Otherwise, we prompt the user for a URL to download the cache therefrom 
            (compressed or not). It saves it in the working directory with the name
            cache_file_name
    
    If we are in dev_mode, additionally carry out the following tasks
      - Try and load the temporary cache file, if it exists, and update the main cache
        with it
      - If the delay_responses parameter has changed (i.e., if delay_responses in the cache
        file is not equal to delay_responses_new), it is overwritten in the cache file
    
    It returns a tuple with two entries
      - The value of delay_responses to use
      - The cache
    '''

    # Load the existing cache
    # -----------------------

    # Get the location of the package in case we need it later
    package_location = importlib.resources.files(__package__)

    # Determine the location of the cache file
    if os.path.exists(cache_file_name):
        # First, look in the working directory
        cache_loc = cache_file_name
        
        if verbose:
            print('Cache file found in the working directory.')

    elif os.path.exists(package_location.joinpath(cache_file_name)):
        # If there is no cache file in the working directory, look in the package
        # directory
        cache_loc = package_location.joinpath(cache_file_name)
        
        if verbose:
            print('Cache file found in the package directory.')

    else:
        # If we have neither and we are in dev mode, start with an empty cache; if not
        # ask the user whether they want to download a cache file, and download it to the
        # working directory
        if dev_mode:
            print('No cache file was found; starting with an empty cache. It will be saved in '
                    f'your working directory, with the name {cache_file_name}.')
            cache_loc = cache_file_name
            pickle.dump((delay_responses_new, {}), open(cache_loc, 'wb'))

        else:
            if __package__ == 'cached_openai':
                cache_url = input('No cache file found; please enter a URL to download a cache file from\n')

                try:
                    download_cache(cache_url, cache_file_name)
                    cache_loc = cache_file_name
                    
                    print('Cache file downloaded successfully.')
                except:
                    print('Failed to download cache file')
                    raise
            
            else:
                # Start with an empty cache
                cache_loc = None
        
    # Finally, read the cache file
    if cache_loc:
        delay_responses, cache = load_cache_file(cache_loc)
    else:
        delay_responses = False
        cache = {}
    
    # Update with the temporary cache file, or the new delay_responses value
    # ----------------------------------------------------------------------

    # If we are in dev mode, load any temporary cache file, and update the cache with it; then,
    # re-save the updated cache
    if dev_mode:
        if os.path.exists(temp_cache_file_name):
            if verbose:
                print('Temporary cache file found; using it to update the cache.')
            
            # Load the temporary cache file
            with open(temp_cache_file_name, 'rb') as f:
                while True:
                    # Read the length of the next entry; break if we're reached the end of
                    # the file
                    length_data = f.read(4)
                    if not length_data:
                        break
                    length = struct.unpack('I', length_data)[0]

                    key, value = pickle.loads(f.read(length))

                    cache[key] = value

            # Re-save the cache (note that in dev mode, we always save it uncompressed)
            # Use the new delay_responses value
            pickle.dump((delay_responses_new, cache), open(cache_loc, 'wb'))

            # Delete the temporary cache file
            os.remove(temp_cache_file_name)

            if verbose:
                print('Cache updated and re-saved. Temporary cache file deleted')

        elif delay_responses_new != delay_responses:
            pickle.dump((delay_responses_new, cache), open(cache_loc, 'wb'))

            if verbose:
                print('delay_responses value was changed - cache re-saved')

        # Set the value of delay_responses to the new value
        delay_responses = delay_responses_new

    # Return
    # ------
    return delay_responses, cache