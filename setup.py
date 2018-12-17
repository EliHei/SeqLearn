from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.readlines()[1]

setup(name='seqlearner',
      version='0.0.7',
      description='The multitask learning package for semi-supervised learning on biological sequences',
      long_description=long_description,
      long_description_content_type="text/markdown",
      url='https://github.com/EliHei/SeqLearner',
      author='Elyas Heidari, Mohsen Naghipourfar',
      author_email='almasmadani@gmail.com, mn7697np@gmail.com',
      license='MIT',
      packages=find_packages(),
      zip_safe=False,
      classifiers=[
          "Programming Language :: Python :: 3",
          "License :: OSI Approved :: MIT License",
          "Operating System :: OS Independent",
      ],
      )
