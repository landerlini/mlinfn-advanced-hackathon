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
    __testdir("ex/tests", "tensorflow_env.ipynb", kernel_name=kn)

@pytest.mark.parametrize("kn", ["qml"])
def test_env_quantum(kn):
    __testdir("ex/tests", "quantum_env.ipynb", kernel_name=kn)

@pytest.mark.parametrize("kn", ["cnn-k2", "cnn-k3"])
def test_ex_lhcf_cnn(kn):
    __testdir("ex/tests/lhcf-cnn", "train_and_split.ipynb", kernel_name=kn)
    __testdir("ex/tests/lhcf-cnn", "Network.ipynb", kernel_name=kn)

@pytest.mark.parametrize("kn", ["gan-k2"])
def test_ex_gan_detector(kn):
    __testdir("ex/tests/gan-detector", "Gauss_smearing_GAN.ipynb", kernel_name=kn)

@pytest.mark.parametrize("nb", ["sMRI_fMRI_sep", "Joint_Fusion"])
@pytest.mark.parametrize("kn", ["ai4ni"])
def test_ex_asd_diagnosis(nb, kn):
    __testdir("ex/tests/asd-diagnosis", f"{nb}.ipynb", kernel_name=kn)

@pytest.mark.parametrize("nb", ["QClassifier_*", "QAE_*", "QUBO_*"])
@pytest.mark.parametrize("kn", ["qml"])
def test_ex_quantum_ml(nb, kn):
    __testdir("ex/tests/quantum-ml", nb, kernel_name=kn)
