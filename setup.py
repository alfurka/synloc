import os
from setuptools import setup, find_packages

this_dir = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(this_dir, "README.md"), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="synloc",
    version="0.0.1",
    author="Ali Furkan Kalay",
    author_email="alfurka@gmail.com",
    url="https://github.com/alfurka/synloc",  
    description="A Python package to create synthetic data from a locally and sequentially estimated distributions.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "pandas",
        "numpy",
        "scipy",
        "matplotlib",
        "synthia",
        "sklearn",
        "tqdm"
    ],
    packages=find_packages(),
)