from setuptools import setup, find_packages

setup(name='nlp2020',
      version='0.0.1',
      install_requires=['gym','tqdm','torchtext==0.5.0', "numpy","nltk"],
      packages=find_packages()
)


