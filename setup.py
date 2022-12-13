# copyright ################################## #
# This file is part of the Xheadtail Package.  #
# Copyright (c) CERN, 2021.                    #
# ############################################ #

from setuptools import setup
from pathlib import Path

version_file = Path(__file__).parent / "xheadtail/_version.py"
dd = {}
with open(version_file.absolute(), "r") as fp:
    exec(fp.read(), dd)
__version__ = dd["__version__"]


setup(
    name="xheadtail",
    version=__version__,
    description="In-memory serialization and code generator for CPU and GPU",
    long_description="In-memory serialization and code generator for CPU and GPU",
    author="Lotta Mether",
    author_email="lotta.mether@cern.ch",
    url="https://xsuite.readthedocs.io/",
    python_requires=">=3.7",
    setup_requires=[],
    install_requires=["numpy", "cffi"],
    packages=["xheadtail"],
    license="Apache 2.0",
    download_url="https://pypi.python.org/pypi/xheadtail",
    project_urls={
        "Bug Tracker": "https://github.com/xsuite/xsuite/issues",
        "Documentation": "https://xsuite.readthedocs.io/",
        "Source Code": "https://github.com/xsuite/xheadtail",
    },
    extras_require={
        "tests": ["pytest"],
    },
)
