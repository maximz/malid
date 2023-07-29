#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

# requirements = ["numpy", "matplotlib", "pandas", "seaborn", 'Click>=7.0', 'IPython', 'pandas', 'nbexec']
# setup_requirements = ["pytest-runner", ]
# test_requirements = ["pytest>=3", "scanpy", "python-igraph", "louvain", "pytest-mpl"]

setup(
    python_requires=">=3.9",
    # install_requires=requirements,
    # include_package_data=True,
    name="malid",
    packages=find_packages(include=["malid", "malid.*"]),
    # setup_requires=setup_requirements,
    # test_suite="tests",
    # tests_require=test_requirements,
    version="0.0.1",
    # zip_safe=False,
)
