#!/usr/bin/env python
from setuptools import setup, find_packages

setup(name='factextraction',
      version='1.0',
      description='Framework to collect syntactic and semantic indexes from Wikidata',
      url='#',
      author='Nicolas Gutehrlé',
      author_email='name@example.com',
      packages=['wikidataparsing', 'utils', 'processing', 'ner','model'],
      # exclude_package_data={'': ['.git']},
      # packages=find_packages(exclude=("*.git*",)),
    #   license='MIT',
      # packages=find_packages(where='main'),
      # package_dir={"":"main"},

      zip_safe=False)