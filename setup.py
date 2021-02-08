from setuptools import setup

setup(name='grb_bilby',
      version='0.1',
      description='GRB analysis pipeline with bilby',
      url='https://github.com/nikhil-sarin/grb_bilby',
      author='Nikhil Sarin, Moritz Huebner',
      author_email='nikhil.sarin@monash.edu',
      license='MIT',
      packages=['grb_bilby'],
      package_dir={'grb_bilby': 'grb_bilby'},
      package_data={'grb_bilby': ['priors/*', 'tables/*'], 'grb_bilby.fluxtolum': ['data/*']},
      zip_safe=False)
