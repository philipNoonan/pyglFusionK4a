#!/usr/bin/env python

"""The setup script."""

import os
from setuptools import setup, find_packages

requirements = [
    'glfw',
    'PyOpenGL',
    'opencv-python',
    'imgui',
    'pyk4a'
]

setup(
    author='PN',
    author_email='philip.noonan@kcl.ac.uk',
    python_requires='>=3.6',
    entry_points={
        'console_scripts': [
            'pyglFusionK4A=pyglFusionK4A.pyglFusionK4A:main',
        ],
    },
    install_requires=requirements,
    include_package_data=True,
    name='pyglFusionK4A',
    packages=find_packages(include=['pyglFusionK4A', 'pyglFusionK4A.*']),
    setup_requires=[],
    test_suite='tests',
    tests_require=[],
    url='https://github.com/philipNoonan/pyglFusionK4A',
    version='0.1.0',
    zip_safe=False,
)
