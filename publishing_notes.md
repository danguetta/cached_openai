Version is in toml file

# To publish a new version of the pacakge

Ensure build is installed

```pip install --upgrade build```

Do the build in the directory with `pyproject.toml`

```python -m build```

Ensure twine is installed

```pip install --upgrade twine```

## Test publishing
Get an API token from (select "entire account" as the scope)

```https://test.pypi.org/manage/account/#api-tokens```

Publish

```python3 -m twine upload --repository testpypi dist/*```

Test install (don't use dependencies because test pypi might not have the same packages available)

```pip install --index-url https://test.pypi.org/simple/ --no-deps cached_openai```

## Real publishing
Get an API token from (select "entire account" as the scope)

```https://pypi.org/manage/account/#api-tokens```

Publish

```python3 -m twine upload dist/*```

## To publish a specific sub-package

See publish_alt.sh