from setuptools import setup

setup(name='ir_ldr',
      version='0.1.0',
      description='The python package to deal with infrared LDR and Teff.',
      url='https://github.com/MingjieJian/ir_ldr',
      author='Mingjie Jian',
      author_email='ssaajianmingjie@gmail.com',
      license='MIT',
      packages=['ir_ldr'],
      install_requires=[
          'numpy',
          'pandas',
          'matplotlib',
          'scipy',
      ],
      zip_safe=False)
