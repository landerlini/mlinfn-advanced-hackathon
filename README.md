# Material for the AI-INFN Advanced Hackathon

AI-INFN, as evolution of the ML-INFN initiative, collects and coordinate the efforts 
on the development and deployment of Artificial Intelligence technologies relevant 
to INFN research. As part of our programme, we organize training events to discuss 
base and advanced Machine Learning topics with time to go through the code.
We call them _hackathons_.

The [first](https://agenda.infn.it/event/25855), 
[second](https://agenda.infn.it/event/28565), and 
[fourth](https://agenda.infn.it/event/35607) ML-INFN hackathons were targeting
Machine Learning beginners. The material used for those events is available in 
[another GitHub repository](https://github.com/tommasoboccali/ml_infn_hackBase).

The [third](https://agenda.infn.it/event/32568) and 
[fifth](https://agenda.infn.it/event/37650) ML-INFN hackathons were targeting 
advanced users, as well as this [first](https://agenda.infn.it/event/43129) 
AI-INFN hackathon and the related materials are collected in this repository.

## Structure of the repository
Contents is organized per topic in different folders. 
When documentation beyond the Jupyter notebook is needed, a README.md file is 
included in the sub-directory.

### Contents
* `ex`: material for the hackathon exercises
  * [`lhcf-cnn`](./ex/lhcf-cnn): Use of a multidimensional CNN for particle 
  identification in the LHCf experiment
  * [`gan-detector`](./ex/gan-detector): Generative Adversarial Networks as 
  a tool to unfold detector effects
  * [`asd-diagnosis`](./ex/asd-diagnosis): Autism Spectrum Disorders (ASD) 
  diagnosis using structural and functional Magnetic Resonance Imaging and 
  Radiomics
  * [`quantum-ml`](./ex/quantum-ml): Quantum Machine Learning applications: 
  classification, anomaly detection and QUBO problems

## Automated testing
Tests on the notebooks are run frequently on the different setups being prepared
for the hackathon event.

Run all tests with:
```bash
python3 -m pytest tests/test_notebooks.py -v --durations=0
```

### Latest results

#### CNAF - A100 with MIG (2024-11-24)
```
85.10s call     tests/test_notebooks.py::test_ex_asd_diagnosis[ai4ni-sMRI_fMRI_sep]
78.64s call     tests/test_notebooks.py::test_ex_asd_diagnosis[ai4ni-Joint_Fusion]
74.39s call     tests/test_notebooks.py::test_ex_quantum_ml[qml-QClassifier_*]
74.38s call     tests/test_notebooks.py::test_ex_gan_detector[gan-k2]
59.21s call     tests/test_notebooks.py::test_ex_lhcf_cnn[cnn-k3]
48.38s call     tests/test_notebooks.py::test_ex_lhcf_cnn[cnn-k2]
28.07s call     tests/test_notebooks.py::test_ex_quantum_ml[qml-QAE_*]
12.98s call     tests/test_notebooks.py::test_ex_quantum_ml[qml-QUBO_*]
10.03s call     tests/test_notebooks.py::test_env_tensorflow[gan-k3]
10.03s call     tests/test_notebooks.py::test_env_quantum[qml]
9.00s call     tests/test_notebooks.py::test_env_tensorflow[cnn-k2]
8.53s call     tests/test_notebooks.py::test_env_tensorflow[cnn-k3]
7.67s call     tests/test_notebooks.py::test_env_tensorflow[gan-k2]
7.56s call     tests/test_notebooks.py::test_env_tensorflow[ai4ni]
```

#### ReCaS - A100 with MIG (2024-11-16)
```
90.53s call     tests/test_notebooks.py::test_ex_quantum_ml[qml-QClassifier_*]
74.62s call     tests/test_notebooks.py::test_ex_asd_diagnosis[ai4ni-sMRI_fMRI_sep]
72.61s call     tests/test_notebooks.py::test_ex_asd_diagnosis[ai4ni-Joint_Fusion]
64.08s call     tests/test_notebooks.py::test_ex_gan_detector[gan-k2]
57.02s call     tests/test_notebooks.py::test_ex_lhcf_cnn[cnn-k2]
56.07s call     tests/test_notebooks.py::test_ex_lhcf_cnn[cnn-k3]
30.02s call     tests/test_notebooks.py::test_ex_quantum_ml[qml-QAE_*]
13.12s call     tests/test_notebooks.py::test_ex_quantum_ml[qml-QUBO_*]
9.03s call     tests/test_notebooks.py::test_env_quantum[qml]
6.72s call     tests/test_notebooks.py::test_env_tensorflow[gan-k3]
6.59s call     tests/test_notebooks.py::test_env_tensorflow[cnn-k3]
6.02s call     tests/test_notebooks.py::test_env_tensorflow[gan-k2]
5.96s call     tests/test_notebooks.py::test_env_tensorflow[ai4ni]
5.76s call     tests/test_notebooks.py::test_env_tensorflow[cnn-k2]
```

## License
Code is released under OSI-approved [MIT license](./LICENSE).

The documentation provided in the form of Jupyter notebooks is 
released under [CC-BY-NC-SA](./CC-BY-NC-SA-4.0) license.
