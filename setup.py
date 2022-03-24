from setuptools import setup, find_packages

# To use a consistent encoding
from codecs import open
from os import path

# The directory containing this file
HERE = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(HERE, 'README.md'), 'r', encoding='utf-8') as f:
    long_description = f.read()


with open(path.join(HERE, 'genz_tokenize', 'requires.txt'), 'r', encoding='utf-8') as f:
    reqs = [i.strip() for i in f.readlines()]

# This call to setup() does all the work
setup(
    name="genz-tokenize",
    version="1.1.7",
    description="""Vietnamese tokenization, preprocess and models NLP""",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/nghiemIUH/genz-tokenize",
    author="Van Nghiem",
    author_email="vannghiem848@gmail.com",
    license="MIT",
    classifiers=[
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Operating System :: OS Independent"
    ],
    packages=["genz_tokenize"],
    package_data={'genz_tokenize': [
        'data/vocab.txt', 'data/bpe.codes', 'requires.txt',
        'data/emb_1.pkl', 'data/emb_2.pkl', 'data/emb_3.pkl']},
    include_package_data=True,
    install_requires=reqs
)
