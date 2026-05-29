# Make the main entrypoints into the package available at the top level of the package
from .main import OpenAI, AsyncOpenAI, DEV_MODE

# Make the materialization functions available at the top level of the package, if
# we're in dev mode
if DEV_MODE:
    from .main import materialize

# Remove DEV_MODE from the namespace
del DEV_MODE

# Load openai.pydantic_function_tool, which we'll need
from openai import pydantic_function_tool

# Create a utility function to open the containing folder the notebook is in
def show_file_location():
    from pathlib import Path
    import platform
    import subprocess
    import os

    path = Path.cwd()

    if platform.system() == "Windows":
        os.startfile(path)
    elif platform.system() == "Darwin":
        subprocess.run(["open", path])
    elif platform.system() == "Linux":
        from IPython.display import Javascript, display

        display(Javascript("""
        window.open(
        window.location.href.replace('/notebooks/', '/tree/').replace(/\\/[^\\/]*$/, ''),
        '_blank'
        )
        """))