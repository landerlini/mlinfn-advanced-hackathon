import os
import pytest
import nbformat
from glob import glob
from nbconvert.preprocessors import ExecutePreprocessor, CellExecutionError
from working_dir import working_dir


def __testdir(a_dir: str, file_pattern: str = "*.ipynb", kernel_name: str = "python3"):
    with working_dir(a_dir):
      filenames = glob(file_pattern)
      for filename in [f for f in filenames if '.TESTED.' not in f]:
          with open(filename) as f:
              nb = nbformat.read(f, as_version=4)

          ep = ExecutePreprocessor(kernel_name=kernel_name)

          ep.preprocess(nb)

          with open(filename.replace(".ipynb", ".TESTED.ipynb"), 'w') as f:
              nbformat.write(nb, f)


@pytest.mark.parametrize("kn", ["cnn-k2", "cnn-k3", "gan-k2", "gan-k3", "ai4ni"])
def test_env_tensorflow(kn):
    return __testdir("ex/tests", "tensorflow_env.ipynb", kernel_name=kn)

@pytest.mark.parametrize("kn", ["qml"])
def test_env_quantum(kn):
    return __testdir("ex/tests", "quantum_env.ipynb", kernel_name=kn)

@pytest.mark.parametrize("kn", ["cnn-k2", "cnn-k3"])
def test_ex_lhcf_cnn(kn):
    return __testdir("ex/tests/lhcf-cnn", kernel_name=kn)

@pytest.mark.parametrize("kn", ["gan-k2"])
def test_ex_gan_detector(kn):
    return __testdir("ex/tests/gan-detector", kernel_name=kn)

@pytest.mark.parametrize("kn", ["ai4ni"])
def test_ex_asd_diagnosis(kn):
    return __testdir("ex/tests/asd-diagnosis", kernel_name=kn)

@pytest.mark.parametrize("kn", ["qml"])
def test_ex_quantum_ml(kn):
    return __testdir("ex/tests/quantum-ml", kernel_name=kn)
