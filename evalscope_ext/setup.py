"""
Install the evalscope_ext package in editable mode:

    cd <evalscope-fork-root>
    pip install -e .              # installs evalscope itself
    pip install -e ./evalscope_ext   # installs this extension

Or from the fork root if evalscope_ext/ is a sibling of evalscope/:

    pip install -e path/to/evalscope_ext
"""

from setuptools import setup, find_packages

setup(
    name="evalscope-ext",
    version="0.1.0",
    description="Cerebras benchmark pruning extension for evalscope",
    packages=find_packages(),
    package_data={
        "evalscope_ext": [
            "pruning/indices/*.json",
        ],
    },
    python_requires=">=3.9",
    install_requires=[
        "evalscope>=1.8.0",
    ],
)