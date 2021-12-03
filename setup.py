import setuptools

setuptools.setup(
    name='augdino',
    version='0.0.6',
    author='Daniel Leong',
    author_email='daniel.leongsj@gmail.com',
    description='A module for torch-based audio data augmentations',
    long_description='A module for torch-based audio data augmentations',
    long_description_content_type="text/markdown",
    url='https://github.com/DinoHub/augdino',
    project_urls = {
        "Bug Tracker": "https://github.com/DinoHub/augdino/issues"
    },
    license='MIT',
    packages=setuptools.find_packages(),
    install_requires=['torch>=1.8.1,<1.9', 'torchaudio>=0.8.1,<0.9', 'librosa==0.8.1', 'julius>=0.2.6', 'primePy>=1.3'],
)