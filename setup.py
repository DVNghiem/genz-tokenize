from setuptools import setup
import setuptools

# To use a consistent encoding
from codecs import open
from os import path

# The directory containing this file
HERE = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(HERE, 'README.md'), 'r', encoding='utf-8') as f:
    long_description = f.read()


reqs = [
    'numpy',
    'transformers',
    'tensorflow',
]


setup(
    name="genz-tokenize",
    version="1.1.8",
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
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent"
    ],
    install_requires=reqs,
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.7",
    package_data={'genz_tokenize': ['data/*']},
    include_package_data=True,

)
