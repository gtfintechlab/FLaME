# setup.py
from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

packages = find_packages(where="src")
print("Found packages:", packages)

setup(
    name="ferrari",
    version="0.0.1",
    description="FERRArI - Financial Economics Reasoning Refinement for Artificial Intelligence",
    author="Glenn Matlin",
    author_email="gmatlin3@gatech.edu",
    url="https://github.com/gtfintechlab/ferrari",  # Replace with your project's URL
    packages=packages,
    package_dir={"": "src"},
    include_package_data=True,
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
