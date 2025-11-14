"""
Setup script for redback-example-plugin.

This demonstrates how to create a plugin package for Redback.
"""

from setuptools import setup, find_packages

setup(
    name='redback-example-plugin',
    version='0.1.0',
    description='Example plugin package for Redback demonstrating the plugin system',
    author='Redback Team',
    packages=find_packages(),
    install_requires=[
        'redback>=1.12.0',
        'numpy',
    ],
    entry_points={
        'redback.model.modules': [
            'example_models = redback_example_plugin.models',
        ],
    },
    python_requires='>=3.10',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
    ],
)
