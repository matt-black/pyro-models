import os
from setuptools import setup

#def read(fname):
#    return open(os.path.join(os.path.dirname(___file__), fname)).read()


setup(
    name="pyromodels",
    version="0.0.1",
    author="Matt Black",
    author_email="mb46@princeton.edu",
    description="statistical models, in pyro",
    #long_description=read("README.md"),
    packages=["pyromodels"],
)
