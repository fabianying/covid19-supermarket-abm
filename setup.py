import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="covid19-supermarket-abm",
    version="0.0.2",
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

    # install_requires=[
    #     'pandas',
    #     'numpy',
    #     'simpy',
    #     'matplotlib',
    #     'networkx',
    #     'tqdm',
    #     'pyarrow'
    # ],

)