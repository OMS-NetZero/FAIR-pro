# -*- coding: utf-8 -*-
# @Author: Zebedee Nicholls
# @Date:   2017-04-10 13:42:11
# @Last Modified by:   Zebedee Nicholls
# @Last Modified time: 2017-04-16 19:15:00

from setuptools import setup
from setuptools import find_packages

# README #
def readme():
    with open('README.md') as f:
        return f.read()

# VERSION #
import re
VERSIONFILE="_version.py"
verstrline = open(VERSIONFILE, "rt").read()
VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
mo = re.search(VSRE, verstrline, re.M)
if mo:
    verstr = mo.group(1)
else:
    raise RuntimeError("Unable to find version string in %s." % (VERSIONFILE,))

setup(name='fair',
      version=verstr,
      description='Python package to perform calculations with the FAIR simple climate model',
      long_description=readme(),
      # classifiers=[
      #   'Development Status :: 3 - Alpha',
      #   'License :: OSI Approved :: MIT License',
      #   'Programming Language :: Python :: 2.7',
      #   'Topic :: Text Processing :: Linguistic',
      # ],
      keywords='simple climate model temperature response carbon cycle',
      url='https://github.com/OMS-NetZero/FAIR',
      author='OMS-NetZero, Chris Smith, Richard Millar, Zebedee Nicholls, Myles Allen',
      author_email='c.j.smith1@leeds.ac.uk/richard.millar@physics.ox.ac.uk',
      license='Apache 2.0',
      packages=find_packages(exclude=['tests*']),
      install_requires=[
          'numpy',
          'scipy',
      ],
      # include_package_data=True,
      # zip_safe=False,
      setup_requires=['pytest-runner'],
      # tests_require=['pytest','numpy'],
)
