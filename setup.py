"""
open_combind
Open-source docking pipeline leveraging pairwise statistics
"""
import sys
from setuptools import setup, find_packages
import versioneer

short_description = "Open-source docking pipeline leveraging pairwise statistics".split("\n")[0]

# from https://github.com/pytest-dev/pytest-runner#conditional-requirement
needs_pytest = {'pytest', 'test', 'ptr'}.intersection(sys.argv)
pytest_runner = ['pytest-runner'] if needs_pytest else []

try:
    with open("README.md", "r") as handle:
        long_description = handle.read()
except:
    long_description = None


setup(
    # Self-descriptive entries which should always be present
    name='open_combind',
    author='Andrew McNutt',
    author_email='anm329@pitt.edu',
    description=short_description,
    long_description=long_description,
    long_description_content_type="text/markdown",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    license='Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported',

    # Which Python importable modules should be included when your package is installed
    # Handled automatically by setuptools. Use 'exclude' to prevent some specific
    # subpackage(s) from being added, if needed
    packages=find_packages(),

    # Optional include package data to ship with your package
    # Customize MANIFEST.in if the general case does not suit your needs
    # Comment out this line to prevent the files from being packaged with your software
    include_package_data=True,
    package_data={
        "statistics_data": ["stats_data/default/"],
        "custom_atom_types": ["dock/crossdock_atom_types.txt"],
    },

    # Allows `setup.py test` to work correctly with pytest
    setup_requires=[] + pytest_runner,

    # Additional entries you may want simply uncomment the lines you want and fill in the data
    url='https://github.com/drewnutt/open_combind',  # Website
    install_requires=["pandas","numpy","click","plumbum","tqdm",
                        "ProDy>=2.0,!=2.4.0","rdkit-pypi","requests"],              # Required packages, pulls from pip if needed; do not use for Conda deployment
    
    entry_points={
        "console_scripts": [
            "open_combind=open_combind.cli.cli:cli",
        ]
    },
    # platforms=['Linux',
    #            'Mac OS-X',
    #            'Unix',
    #            'Windows'],            # Valid platforms your code works on, adjust to your flavor
    # python_requires=">=3.5",          # Python version restrictions

    # Manual control if final package is compressible or not, set False to prevent the .egg from being made
    # zip_safe=False,

)
