from setuptools import setup, find_packages

setup(
    name="lab3",
    version="0.0.1",
    packages=find_packages(),
    license="MIT",
    description="lab3",
    install_requires=["pytorch-model-utils @ git+https://github.com/bhbbbbb/pytorch-model-utils@fc95bb6"],
    dependency_links=["git+https://github.com/bhbbbbb/pytorch-model-utils@fc95bb6"],
)
