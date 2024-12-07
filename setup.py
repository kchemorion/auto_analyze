"""
jupyterlab_auto_analyze setup
"""
import json
import sys
from pathlib import Path

import setuptools

HERE = Path(__file__).parent.resolve()

# Get the package info from package.json
pkg_json = json.loads((HERE / "package.json").read_bytes())

setuptools.setup(
    name="jupyterlab_auto_analyze",
    packages=setuptools.find_packages(),
    install_requires=[
        "jupyterlab>=4.0.0,<5.0.0",
    ],
    zip_safe=False,
    include_package_data=True,
    python_requires=">=3.8",
)