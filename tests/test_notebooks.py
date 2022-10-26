import os
import nbformat
from glob import glob
from nbconvert.preprocessors import ExecutePreprocessor

ORIG_DIR = os.getcwd()

def __testdir (a_dir: str, file_pattern: str = "*.ipynb"):
    os.chdir(a_dir)
    try:
      filenames = glob(file_pattern)
      for filename in [f for f in filenames if '.TESTED.' not in f]:
          with open(filename) as f:
              nb = nbformat.read(f, as_version=4)

          ep = ExecutePreprocessor(kernel_name='python3')

          ep.preprocess(nb)

          with open(filename.replace(".ipynb", ".TESTED.ipynb"), 'w') as f:
              nbformat.write(nb, f)
    finally:
      os.chdir(ORIG_DIR)


def test_intro_gnn():
  return __testdir('introduction_to_gnns')

def test_transformers():
  return __testdir('ex/gnn_transformers', 'TransformerSG.ipynb')
