from setuptools import setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

requirement_path = f"requirements.txt"
with open(requirement_path) as f:
    install_requires = list(f.read().splitlines())

optional_requirement_path = f"optional_requirements.txt"
with open(optional_requirement_path) as f:
    optional_install_requires = list(f.read().splitlines())

setup(
    name='redback',
    version='1.0.2',
    description='A Bayesian inference pipeline for electromagnetic transients',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/nikhil-sarin/redback',
    author='Nikhil Sarin, Moritz Huebner',
    author_email='nikhil.sarin@su.se',
    license='GNU General Public License v3 (GPLv3)',
    packages=['redback', 'redback.get_data', 'redback.transient', 'redback.transient_models'],
    package_dir={'redback': 'redback', },
    package_data={'redback': ['priors/*', 'tables/*', 'plot_styles/*']},
    install_requires=[
        "numpy<1.26",
        "setuptools",
        "pandas",
        "scipy",
        "selenium",
        "sncosmo",
        "matplotlib",
        "astropy",
        "afterglowpy",
        "extinction",
        "requests",
        "lxml",
        "sphinx-rtd-theme",
        "sphinx-tabs",
        "bilby",
        "regex",
    ],
    extras_require={
        'all': [
            "nestle",
            "sherpa",
            "scikit-learn",
            "PyQt5",
            "lalsuite",
            "kilonova-heating-rate",
            "redback-surrogates",
            "kilonovanet"
        ]
    },
    python_requires=">=3.7",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    zip_safe=False
)
