import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="covid19-supermarket-abm",
    version="0.0.5",
    author="Fabian Ying",
    author_email="fabian.m.ying@gmail.com",
    description='Agent-based model for Covid-19 transmission in supermarkets',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/fabianying/covid19-supermarket-abm",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License (GPL)",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    package_data={'': ['example_data/*']},

    packages=setuptools.find_packages(
        exclude=[
            '*utils_for_paper',
            '*tests'
        ]
    ),

    install_requires=[
        'pandas>=0.24.2',
        'numpy>=1.16.2',
        'simpy>=4.0.1',
        'matplotlib>=3.1.0',
        'networkx>=2.5',
        'tqdm',
        'pyarrow>=0.16.0'
    ],

)