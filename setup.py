from setuptools import setup, find_packages
import os
import sys

setup(
    name="arenahard",
    version="0.1.0",
    author="NM MLR",
    description="Arena Hard",
    url="https://github.com/neuralmagic/arena-hard-auto",
    package_dir={"": "src"},
    packages=find_packages(
        "src", include=["arenahard", "arenahard.*"], exclude=["*.__pycache__.*"]
    ),
    install_requires=[
        "tiktoken",
        "openai",
        "numpy",
        "pandas",
        "shortuuid",
        "tqdm",
        "gradio==5.25.2",
        "plotly",
        "scikit-learn",
        "boto3",
    ],
    python_requires=">=3.7",
)
