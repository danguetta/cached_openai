#!/bin/bash

echo "Which package do you want to compile (eg: 'abf')?"

read package_name

mv dist "dist_old_"
mv "dist_""$package_name" dist

mv "pyproject.toml" "pyproject_old_.toml"
mv "pyproject_""$package_name"".toml" "pyproject.toml"

mv "src/cached_openai" "src/""$package_name""_openai"

python -m build
python3 -m twine upload dist/*

mv "src/""$package_name""_openai" "src/cached_openai"

mv "pyproject.toml" "pyproject_""$package_name"".toml"
mv "pyproject_old_.toml" "pyproject.toml"

mv dist "dist_""$package_name"
mv "dist_old_" dist