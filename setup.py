from setuptools import setup, find_packages

setup(
    name='tctk',
    version='0.1.20',
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    url='https://github.com/nhgritctran/tctk',
    license='GPL-3.0',
    author='Tam Tran',
    author_email='tam.c.tran@outlook.com',
    description='A collection of mini tools.',
    install_requires=[
        "connectorx",
        "google-cloud-bigquery",
        "polars",
        "pyarrow",
        "tableone",
        "tqdm"
    ]
)
