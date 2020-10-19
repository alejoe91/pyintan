from setuptools import setup, find_packages

d = {}
exec(open("pyintan/version.py").read(), None, d)
version = d['version']
long_description = open("README.md").read()

entry_points = None

setup(
    name="pyintan",
    version=version,
    author="Alessio Buccino",
    author_email="alessiob@ifi.uio.no",
    description="Python package for parsing INTAN data.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/alejoe91/pyintan",
    packages=find_packages(),
    entry_points=entry_points,
    include_package_data=True,
    install_requires=[
        'numpy',
        'pyyaml',
        'quantities',
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ]
)
