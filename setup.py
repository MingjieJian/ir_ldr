import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
      name='ir_ldr',
      version='0.2.1',
      description='The python package to deal with infrared LDR and Teff.',
      long_description=long_description,
      long_description_content_type="text/markdown",
      url='https://github.com/MingjieJian/ir_ldr',
      author='Mingjie Jian',
      author_email='ssaajianmingjie@gmail.com',
      classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Framework :: IPython",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
        "Topic :: Scientific/Engineering :: Astronomy"
      ],
      packages=setuptools.find_packages(),
      install_requires=[
          'numpy>=1.17.0',
          'pandas>=0.20.0',
          'matplotlib>=3.0.0',
          'scipy>=1.0.0',
      ],
      include_package_data=True,
      zip_safe=False)
