import os
from setuptools import setup, find_packages

this_dir = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(this_dir, "README.md"), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="synloc",
    version="0.2",
    author="Ali Furkan Kalay",
    author_email="alfurka@gmail.com",
    url="https://github.com/alfurka/synloc",  
    description="A Python package to create synthetic data from a locally estimated distributions.",
    project_urls={
        'Documentation': 'https://alfurka.github.io/synloc/',
    },
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
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
        "scikit-learn",
        "tqdm",
        "k_means_constrained"
    ],
    packages=find_packages(),
    keywords=['copulas', 'distributions','sampling','synthetic-data','oversampling','nonparametric-distributions','semiparametric','nonparametric','knn', 'clustering','k-means','multivariate-distributions'],
)