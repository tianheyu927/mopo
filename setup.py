from distutils.core import setup
from setuptools import find_packages

setup(
    name='mopo',
    packages=find_packages(),
    version='0.0.1',
    description='Model-based Offline Policy Optimization',
    long_description=open('./README.md').read(),
    author='Tianhe Yu',
    author_email='tianheyu@cs.stanford.eu',
    url='https://arxiv.org/abs/2005.13239',
    entry_points={
        'console_scripts': (
            'mopo=softlearning.scripts.console_scripts:main',
            'viskit=mopo.scripts.console_scripts:main'
        )
    },
    requires=(),
    zip_safe=True,
    license='MIT'
)
