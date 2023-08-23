#!/usr/bin/env python

from setuptools import setup, find_packages
import os

# Dynamically read the requirements from requirements.txt
# with open("requirements.txt", "r") as f:
#     requirements = f.read().splitlines()

setup(
    name="substrat-automl",
    version="0.3",
    description="A Python package for automated machine learning tasks with genetic algorithm-based dataset summarization.",
    long_description=open('README.md').read(),
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    long_description_content_type="text/markdown",
    url="https://github.com/YOUR_USERNAME/YOUR_REPO_NAME",  # Adjust with your repo URL
    author="Eyal Elboim",
    author_email="Eyal.Elboim1@gmail.com",  
    install_requires=["pandas>=2,<3",
                    "scipy>=1.10.1,<2",
                    "numpy>=1.25.0,<2",
                    "scikit-learn",
                    "auto-sklearn==0.15.0",
                    "sklearn",
                    "tqdm",
                ],
    python_requires='>=3.8',  
)

