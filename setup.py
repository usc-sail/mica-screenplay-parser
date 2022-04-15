import pathlib
from setuptools import setup

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

# This call to setup() does all the work
setup(
    name="screenplayparser",
    version="0.0.0",
    description="Tag screenplay lines",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/usc-sail/screenplay-parser",
    author="Sabyasachee Baruah",
    author_email="sbaruah@usc.edu",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
    ],
    packages=["reader"],
    include_package_data=True,
    install_requires=["feedparser", "html2text"],
    entry_points={
        "console_scripts": [
            "realpython=reader.__main__:main",
        ]
    },
)
