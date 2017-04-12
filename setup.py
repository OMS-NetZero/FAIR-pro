# -*- coding: utf-8 -*-
# @Author: Zebedee Nicholls
# @Date:   2017-04-10 13:42:11
# @Last Modified by:   Zebedee Nicholls
# @Last Modified time: 2017-04-10 14:05:28

from setuptools import setup
from setuptools import find_packages

def readme():
    with open('README.md') as f:
        return f.read()

setup(name='fair',
      version='0.0.0.dev0',
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
      author='OMS-NetZero, Richard Millar, Zebedee Nicholls',
      author_email='richard.millar@physics.ox.ac.uk',
      # license='Apache',
      packages=find_packages(exclude=['tests*']),
      install_requires=[
          'numpy',
          'scipy',
      ],
      # include_package_data=True,
      # zip_safe=False,
      # test_suite='nose.collector',
      # tests_require=['nose'],
      )