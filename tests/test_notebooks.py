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


def test_intro_gnn():
  return __testdir('introduction_to_gnns')

def test_intro_pytorch():
  return __testdir('introduction_to_pytorch', 'PyTorch_SimpleMLP_Example.ipynb')

def test_ex_gnn():
  return __testdir('ex/gnn_transformers/GNN_IN_JetTagger', 'GNN_Jet_Tagging_IN.ipynb')

def test_transformers():
  return __testdir('ex/gnn_transformers', 'TransformerSG.ipynb')

def test_unet_train_only():
  return __testdir('ex/unet', 'Train_UNET.ipynb')

def test_unet_predict_only():
  return __testdir('ex/unet', 'Predict_UNet.ipynb')

def test_unet_intro():
  return __testdir('ex/unet', 'Lung_Segmentation_on_Chest_X-Ray_images_with_U-Net.ipynb')

def test_unet_generator():
  return __testdir('ex/unet/Data_Generator')

def test_unet_arch():
    with pytest.raises(CellExecutionError):
      return __testdir('ex/unet/UNet_Arch')

def test_unet_loss():
  return __testdir('ex/unet/Loss_Metrics')

def test_DA_ML():
  return __testdir('ex/domain_adaptation', 'Excercise_DA_MLhackathon.ipynb')

def test_DA_ML_SimpleDNN():
  return __testdir('ex/domain_adaptation', 'Excercise_DA_MLhackathon_SimpleDNN.ipynb')

def test_xai():
  return __testdir('ex/xai', 'XAI.ipynb')
