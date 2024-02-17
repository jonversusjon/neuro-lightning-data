from setuptools import setup, find_packages

setup(
    name='neuro-lightning-data',
    version='0.1.0',
    author='Your Name',
    author_email='jonathan.ramirez@ucsf.edu',
    description='A Python package for handling and processing neuroimaging data with PyTorch Lightning.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/jonversusjon/neuro-lightning-data',
    packages=find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
    install_requires=[
        'numpy',
        'torch',
        'torchvision',
        'tqdm',
        'prettytable',
        'pytorch_lightning',
        'Pillow',
        'tifffile',
    ],
    extras_require={
        'dev': [
            'pytest',
            'check-manifest',
            'twine',
        ],
    },
)
