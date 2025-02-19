from setuptools import setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='redback',
    version='1.0.31',
    description='A Bayesian inference pipeline for electromagnetic transients',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/nikhil-sarin/redback',
    author='Nikhil Sarin, Moritz Huebner',
    author_email='nsarin.astro@gmail.com',
    license='GNU General Public License v3 (GPLv3)',
    packages=['redback', 'redback.get_data', 'redback.transient', 'redback.transient_models'],
    package_dir={'redback': 'redback', },
    package_data={'redback': ['priors/*', 'tables/*', 'plot_styles/*']},
    install_requires=[
        "numpy==1.26.0",
        "setuptools",
        "pandas",
        "scipy<1.14.0",
        "selenium",
        "matplotlib",
        "astropy",
        "extinction",
        "requests",
        "lxml",
        "sphinx-rtd-theme",
        "sphinx-tabs",
        "bilby",
        "regex",
        "sncosmo",
        "afterglowpy",
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
            "kilonovanet",
            "astroquery"
        ]
    },
    python_requires=">=3.10",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    zip_safe=False
)
