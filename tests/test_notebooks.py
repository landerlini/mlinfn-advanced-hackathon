import pytest
import os
import nbformat
from glob import glob
from nbconvert.preprocessors import ExecutePreprocessor, CellExecutionError
from working_dir import working_dir

def __testdir (a_dir: str, file_pattern: str = "*.ipynb"):
    with working_dir(a_dir):
      filenames = glob(file_pattern)
      for filename in [f for f in filenames if '.TESTED.' not in f]:
          with open(filename) as f:
              nb = nbformat.read(f, as_version=4)

          ep = ExecutePreprocessor(kernel_name='python3')

          ep.preprocess(nb)

          with open(filename.replace(".ipynb", ".TESTED.ipynb"), 'w') as f:
              nbformat.write(nb, f)




