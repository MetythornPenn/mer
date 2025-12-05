import os
from setuptools import setup

def read(*paths):
    with open(os.path.join(*paths), 'r') as f:
        return f.read()
    
requirements = [
    'torch',
    'torchvision',
    'onnxruntime-gpu',
    'surya-ocr',
    'pillow',
    'huggingface-hub',
    'tqdm',
]

setup(
    name='mer',
    version='1.2.0',
    packages=['mer'],
    url='https://github.com/MetythornPenn/mer.git',
    license='Apache Software License 2.0',
    author = 'Metythorn Penn',
    author_email = 'metythorn@gmail.com',
    keywords='ocr',
    description='Khmer OCR',
    install_requires=requirements,
    long_description=(read('README.md')),
    long_description_content_type='text/markdown',
	classifiers= [
		'Natural Language :: English',
		'License :: OSI Approved :: Apache Software License',
		'Operating System :: OS Independent',
		'Programming Language :: Python :: 3',
	],
)
