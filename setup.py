import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='ruleex',
    version='0.3.0',
    description='Rule extraction methods from neural networks',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/fantamat/ruleex',
    author='Matěj Fanta',
    author_email='fantamat93@gmail.com',
    license='Apache License 2.0',
    packages=['ruleex'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering",
    ]
)
