#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name='rul-custom-models',
    description='Custom pipeline, transformer and estimator for RUL predictions',
    version='0.1',
    author='IBM-Maximo-Predict',
    author_email='pmqcloud@us.ibm.com',
    license='IBM',
    packages=find_packages(),
    install_requires=[
        'pmlib@https://stg.pmi.apm.maximo.ibm.com/ibm/pmi/service/rest/ds/54ecece5/a5la6d40c3brmk6tliskekec054s2s4m0nkvjlq/lib/download?filename=pmlib-1.0.0-master.tar.gz'
    ]
)
