# cached_openai

`cached_openai` is a simple Python library that mimics the `OpenAI` python library, but can draw from a cache instead of sending the request to Open AI.

When it is run in **dev mode**, it caches responses to all requests made from OpenAI, and returns the cached value if the request is made again. When it is run in **production mode**, no new cache entries are created, but if any requests stored in the cache are made again, they are returned therefrom.

It is able to cache several different responses for a single request, to mimic the way the OpenAI API returns different responses when it receives the same query twice. It is also able to handle requests that return images or sound clips.

The user experience with `cached_open` is identical to that with the original `openai` package. Consider, for example, the following piece of code using the "regular" `openai` api:
```
import openai
client = openai.OpenAI(api_key=...)
response = client.chat.completions.create(...)
```

In `cached_openai`, the **only** change required would be to change the first line to `import cached_openai as openai`. That's it - everything else works identically. If the query is cached, the cached version will be returned. If not, the `OpenAI` API will be queries as usual.

The package can be configured to return the cached entry immediately, *or* to replicate the delay that would occur if the request were to be made from OpenAI directly.

The package also works with the async version of the OpenAI API - just use `openai.AsyncOpenAI(...)`.

## Why `cached_openai`?

I designed this package for teaching purposes - when I teach classes that use the OpenAI API, I often provide code to students which they run on their machines before modifying it. This isn't ideal for two reasons
  1. Every student has to pay to run the API requests in the code - there can be hundreds or thousands of these requests, and the costs can add up.
  2. Every student is likely to get different answers out of the API - it makes it difficult to teach the class when everyone is looking at a different answer.

Instead, I can now distributed `cached_openai` with a cache I prepared - students can then run the entire code for free, and get the same answers as I did. They can then modify the code to run their own version, and query OpenAI as usual.

# User manual

This user manual is divided into two parts - preparing a cache, and distributing it.

## Preparing a cache

To prepare a cache, you need to run `cached_openai` in **dev mode**. To do this, set an environment variable called `CACHED_OPENAI_DEV_MODE` to any value before you load the package. I also recommended you set an environment variable called `CACHED_OPENAI_VERBOSE` to any value to print debugging messages as the package runs. The easiest way too do this is in your code, before you import the package

```
import os
os.environ['CACHED_OPENAI_VERBOSE'] = 'True'
os.environ['CACHED_OPENAI_DEV_MODE'] = 'True'
import cached_openai
```

You should then use `cached_openai` exactly as you would `OpenAI` - every request you make will be cached. In dev mode, the package creates three files in your working directory
  - `openai_cache_temp.bin` is a temporary cache file - it is updated every time a call is made to the OpenAI API to store the results of that call. It can be updated very quickly (so as not to slow down the call), but stores data in an inneficient format.
  - `openai_cache.cache` is the main cache file which contains all cached requests in an efficient format (a dictionary). Every time the package is loaded, the data in `openai_cache_temp.bin` is integrated into this permanent cache file, and the temporary file is deleted.
  - `openai_cache_used.txt` stores all the keys of requests that have been made or saved since the file was last created; we'll explain the purpose of this file later.

### Determining delays

The package can be configured to return the cached entry immediately, *or* to replicate the delay that would occur if the request were to be made from OpenAI directly.

By default, the package will be configured to return the result immediately. If you would, instead, like it to replicate the delay that was initially observed when the request was made, simply set an environment variable called `CACHED_OPENAI_DELAY_RESPONSES` to any value before you load the package.

### Repeated requests

When you use `cached_openai` to run a request that has already been run, a new request will not be made to OpenAI - the *cached* result is returned instead, for free.

This does mean, however, that that result will be the same every time. What if you want to store several different responses to a single request? All you need to do is provide a `seed` parameter when you run that same request - if the function has been run with that identical seed before, that result is returned, but if not, the query is run again against the OpenAI API. If the underlying OpenAI API function has a seed parameter (eg: `chat.completions.create`), the seed is passed to that function, and if not (eg: `audio.speech.create`), it is stripped before it is sent.

When the request is next called *without* a seed, it will return each of the seeded responses in succession, which will make it look like it is returning different results every time, just like the original API. Here's an example that might clarify this, using chat completions:

  - **Input**: please give me a random number. **Output**: 42
    - *This is the first time the request has been made, so it is sent to the OpenAI API.*\
  - **Input**: please give me a random number (`seed=1`). **Output**: 25
    - *This is the first time the request has been made with this seed, so it is sent to the OpenAI API.*
  - **Input**: please give me a random number (`seed=2`). **Output**: 126
    - *This is the first time the request has been made with this seed, so it is sent to the OpenAI API.*
  - **Input**: please give me a random number (`seed=2`). **Output**: 126
    - *This request has been made with this seed before, the cached result is returned*
  - **Input**: please give me a random number. **Output**: 42
    - *This request has been made before and is being called again for the first time; return the first saved response.*
  - **Input**: please give me a random number. **Output**: 25
    - *This request has been made before and is being called again for the second time; return the second saved response.*
  - **Input**: please give me a random number. **Output**: 126
    - *This request has been made before and is being called again for the third time; return the third saved response.*
  - **Input**: please give me a random number. **Output**: 42
    - *This request has been made before and is being called again for the four time; there is no fourth response stored, so loop back to the first stored answer.*

### Images

Image requests that return image URLs require specific handling, because OpenAI does not keep these image URLs active forever - they are deleted a few hours after the request is made. Thus, simply storing the original response with the original URL would return a broken link when the cached response was returned.

To solve this problem, `cached_openai` downloads the image from the URL before returning the result, and stores it in the cache. When the same request is later called and retrieved from the cache, the image is extracted into an `image` folder in the working directory, and the URLs in the response are replaced with local URLs pointing to those extracted images.

### Audio

Audio requests suffer from a similar problem - OpenAI returns a file stream which then needs to be downloaded. `cached_openai` handles this seamlessly by downloading the file at request time, saving it in the cache, and simulating a file stream when the request is called again and needs to be retrieved from the cache.

## Distributing the cache

Once you have created the cache for your class, there are three different ways to distribute it to students
  - Get students to install `cached_openai` directly from pypi, and distribute a cache file that they can put in their working directory containing the cache data; you can also publish this file to a URL and `cached_openai` will ask them for a URL from which to download (and potentially decompress) the file.
  - Create your own package on pypi that forks `cached_openai` and includes your own cache file in it (note that this only works if your cache file is small enough to be uploaded to pypi).
  - Create a self-contained `.py` file that contains not only the `cached_openai` code, but also the cache itself, embedded in the file. You can then simply distribute this `.py` file as a "batteries included" package.

In all three cases, the first step is to call the `materialize` function, which will package the cache for you; see the function docstring for details.