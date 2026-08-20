from setuptools import find_packages, setup

setup(
    name="pypsds-gamma",
    version="0.9.0",
    description="CPU/RAM-oriented PS/DS InSAR processing for GAMMA RSLC stacks",
    packages=find_packages(include=["pypsds", "pypsds.*"]),
    python_requires=">=3.11",
)
