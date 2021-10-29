from setuptools import setup, find_packages

setup(
  name = 'synthetic-galaxy-distance',
  packages = find_packages(),
  version = '0.0.1',
  license= 'AGPLv3.0',
  description = 'Synthetic Galaxy Distance',
  author = 'Mike Smith',
  author_email = 'mike@mjjsmith.com',
  url = 'https://github.com/Smith42/SGD',
  keywords = [
    'astronomy',
  ],
  install_requires=[
    'scipy',
    'numpy',
    'glob',
    'tqdm',
    'argparse'
  ],
  classifiers=[
    'License :: OSI Approved :: AGPL License',
    'Programming Language :: Python :: 3',
    'Operating System :: OS'
  ],
)
