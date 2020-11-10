#!/usr/bin/env python3

from setuptools import setup, find_packages


console_scripts = []

setup(
    name="ion_phys",
    version=0,
    description="Ion Physics tool kit",
    entry_points={
        "console_scripts": console_scripts,
    },
    packages=find_packages(),
)
