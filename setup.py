from setuptools import setup, find_packages

setup(
    name="contradiction",
    version="2.0",
    packages=find_packages(where="src"),
    package_dir = {"":"src"},
    include_package_data=True
)