from setuptools import setup, find_packages

setup(name='deepar',
      version='0.0.1',
      description='DeepAR tensorflow implementation',
      author='Alberto Arrigoni',
      author_email='arrigonialberto86@gmail.com',
      url='https://github.com/arrigonialberto86/deepar/tree/master',
      requires=['tensorflow', 'numpy', 'pandas', 'keras'],
      packages=find_packages()
     )