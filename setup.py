from setuptools import setup

from os import path

HERE = path.dirname(__file__)

with open(path.join(HERE, 'README.md'), 'r') as f:
    long_description = f.read()

with open(path.join(HERE, 'genz_tokenize', 'VERSION'), 'r') as f:
    version = f.read().strip()

with open(path.join(HERE, 'requirements.txt'), 'r') as f:
    install_requires = f.read().split('\n')[:-1]

setup(
    name="genz-tokenize",
    version=version,
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
    install_requires=install_requires,
    package_dir={"genz_tokenize": "genz_tokenize"},
    packages=['genz_tokenize'],
    python_requires=">=3.7",
    include_package_data=True,
    zip_safe=False,
    keywords='nlp task',

)
