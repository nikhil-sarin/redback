from setuptools import setup

setup(name='grb_bilby',
      version='0.1',
      description='GRB analysis pipeline with bilby',
      url='https://github.com/nikhil-sarin/grb_bilby',
      author='Nikhil Sarin',
      author_email='nikhil.sarin@monash.edu',
      license='MIT',
      packages=['grb_bilby','grb_bilby.processing','grb_bilby.analysis',
                'grb_bilby.inference','grb_bilby.models'],
      zip_safe=False)
