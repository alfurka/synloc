from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()


setup(
    name="synloc",
    version="0.0.1",
    description="A package to create synthetic data",
    py_modules=["synloc"],
    package_dir={'':'src'},
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
    ],
    python_requires=">=3.8",
    install_requires=[

    ],
    url="https://github.com/alfurka/synloc",
    author="Ali Furkan Kalay",
    author_email="alfurka@gmail.com",

)