from setuptools import setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='redback',
    version='1.12.1',
    description='A Bayesian inference and modelling pipeline for electromagnetic transients',
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
        "numpy",
        "setuptools",
        "pandas",
        "scipy",
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
            "numba",
            "sherpa",
            "george",
            "scikit-learn",
            "PyQt5",
            "lalsuite",
            "kilonova-heating-rate",
            "redback-surrogates",
            "tensorflow",
            "keras",
            "kilonovanet",
            "astroquery",
            "pyphot==1.6.0",
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
