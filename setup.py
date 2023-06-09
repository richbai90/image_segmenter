from setuptools import setup, find_packages

setup(
    name='image_annotator',
    version='1.0.0',
    author='Rich Baird',
    author_email='rich.baird@utah.edu',
    description='A class for annotating images with bounding boxes using various image processing techniques.',
    long_description='...',
    long_description_content_type='text/markdown',
    url='https://github.com/richbai90/image_annotator',
    packages=find_packages(),
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    python_requires='>=3.6',
)
