from setuptools import setup, find_packages

setup(
    name='3dhistograding',
    version='0.1',
    author='Santeri Rytky',
    author_email='santeri.rytky@oulu.fi',
    packages=find_packages(),
    include_package_data=True,
    license='LICENSE',
    long_description=open('README.md').read(),
)
