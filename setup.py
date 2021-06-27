#!/usr/bin/env python
from setuptools import setup, find_packages
import os


REQUIRED = [
  'click>=7.0',
  'numba>=0.51.2',
  'matplotlib>=3.3.2',
  'mattflow>=1.3.4',
  'numpy>=1.18.5',
  'pandas>=1.1.3',
  'pillow',
  'scipy>=1.5.3',
  'tensorflow>=2.3.0',
  'scikit-learn>=0.24.2'
]

EXTRAS = {
  'gpu': ['cudnn==7.6.5', 'cudatoolkit==10.1.243'],
  'test': ['pytest>=5.4.2'],
  'video': ['ffmpeg>=4.2.4']
}

here = os.path.abspath(os.path.dirname(__file__))

# pull info from __init__.py
about = {}
with open(os.path.join(here, 'smartflow', '__init__.py'), 'r') as fr:
  exec(fr.read(), about)

# assign long_description with the README.md content
try:
  with open(os.path.join(here, 'README.md'), 'r') as fr:
    long_description = fr.read()
except FileNotFoundError:
  long_description = about['__description__']


setup(
  name=about['__name__'],
  version=about['__version__'],
  author=about['__author__'],
  author_email=about['__author_email__'],
  licence='GNU GPL3',
  description=about['__description__'],
  long_description=long_description,
  long_description_content_type='text/markdown',
  url=about['__url__'],
  packages=find_packages(exclude=["tests", "*.tests", "*.tests.*", "tests.*"]),
  install_requires=REQUIRED,
  extras_require=EXTRAS,
  include_package_data=True,
  classifiers=[
    'Programming Language :: Python :: 3',
    'Natural Language :: English',
    'License :: OSI Approved :: GNU Lesser General Public License v3 or later (LGPLv3+)',
    'Operating System :: OS Independent',
  ],
  entry_points={
    "console_scripts": [
      "smartflow=smartflow.__main__:main",
    ]
  },
)
