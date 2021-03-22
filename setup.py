from setuptools import setup

setup(name='redback',
      version='0.2',
      description='A Bayesian inference pipeline for electromagnetic transients',
      url='https://github.com/nikhil-sarin/redback',
      author='Nikhil Sarin, Moritz Huebner',
      author_email='nikhil.sarin@monash.edu',
      license='MIT',
      packages=['redback'],
      package_dir={'redback': 'redback'},
      package_data={'redback': ['priors/*', 'tables/*'], 'redback.fluxtolum': ['data/*']},
      zip_safe=False)
