# Cached OpenAI

This repo contains a very simple library that supports a cached version of the OpenAI Python library.

When the library is imported, it looks for a `open_cache.pkl.gz` file in the working directory; this contains a record of previous calls to the OpenAI API and their results.

Every time a function is called, it checks for a result in the cache - if it exists, it returns it for free rather than hitting the API again.

The library is extremely lightweight, and works with *any* function under `openai.OpenAI()` or `openai.AsyncOpenAI()` of whatever version of the `openai` package is currently installed - it does not rely on any specific version. The only restriction is that functions must be called using keyword arguments only (i.e., call a function using `f(x=1, y=2)` instead of `f(1,2)`).

Using it is as simple as switching the import line; instead of

```
import openai
```

ensure `cached_openai.py` is in your working directory and use

```
import cached_openai
```

Then, create a client using `client = cached_openai.OpenAI(api_key='api_key_here')`. Finally, use the client exactly as you would have before (eg: `client.chat.completions.create(...)`)


## Creating a cache

By default, the cache is *not* updated every time a new call is made.

To create a cache, create the client using

```
client = cached_openai.OpenAI(api_key='api_key_here', update_cache=True)
```

Any new call will then be saved in the cache in memory.

Finally, to save this cache to a file, call

```
client.save_cache()
```

This will save a `open_cache.pkl.gz` containing the current cache in memory, which will be loaded every time the package is imported.

### Seed handling

If an OpenAI function is called with a seed while creating the cache, two behaviors are supported:
  - By default, the result will be cached with the seed - in other words, the package will only retrieve the cached entry if the function is called again with the same seed.
  - If the client is created with `strip_seed=True` (i.e. `cached_openai.OpenAI(..., strip_seed=True)`), the result will be saved in the cache *twice* - once with the seed in question, and once without the seed. In the future, if users call the function without any seed specified, that response will also be returned. This is useful in instances where you'd like a specific seeded response to be returned when the function is called with certain values.