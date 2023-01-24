from setuptools import setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(name='redback',
      version='0.3.1',
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
      python_requires=">=3.7",
      classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",],
      zip_safe=False)
