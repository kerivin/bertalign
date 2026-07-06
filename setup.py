from setuptools import setup, find_packages

setup(
    name='Bertalign',
    version='0.1.0',
    url='https://github.com/bfsujason/bertalign',
    description='An automatic mulitlingual sentence aligner.',
    packages=find_packages(),    
    install_requires=[
        "numba>=0.65.1",
        "faiss-gpu>=1.14.3",
        "googletrans>=4.0.2",
        "sentence-splitter>=1.4",
        "sentence-transformers>=5.6.0",
    ],
)