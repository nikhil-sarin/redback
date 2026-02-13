from setuptools import setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='redback',
    version='1.14.0',
    description='A Bayesian inference and modelling pipeline for electromagnetic transients',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/nikhil-sarin/redback',
    author='Nikhil Sarin, Moritz Huebner',
    author_email='nsarin.astro@gmail.com',
    license='GNU General Public License v3 (GPLv3)',
    packages=[
        'redback',
        'redback.get_data',
        'redback.spectral',
        'redback.transient',
        'redback.transient_models',
        'redback.transient_models.afterglow_models',
    ],
    package_dir={'redback': 'redback', },
    package_data={'redback': ['priors/*', 'tables/*', 'tables/xsect/*', 'plot_styles/*']},
    install_requires=[
        "numpy",
        "numba",
        "setuptools",
        "pandas",
        "scipy",
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
            "swifttools"
        ]
    },
    entry_points={
        # Example of how to register model modules as plugins
        # External packages can use these entry point groups to register their models
        # 'redback.model.modules': [
        #     # Format: 'plugin_name = package.module'
        #     # 'example_models = redback_example_plugin.models',
        # ],
        # 'redback.model.base_modules': [
        #     # Base models used as building blocks
        #     # 'example_base_models = redback_example_plugin.base_models',
        # ],
    },
    python_requires=">=3.10",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    zip_safe=False
)
