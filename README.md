# Material for the ML INFN Advanced Hackathon

ML-INFN is the INFN initiative to coordinate efforts on the development and 
deployment of Machine Learning algorithms across its research lines. 
As part of our programme, we organize training events to discuss base and advanced 
Machine Learning topics with time to go through the code. We call them hackathons.

The [first](https://agenda.infn.it/event/25855/overview) and 
[second](https://agenda.infn.it/event/28565/) hackathons were targeting
machine-learning beginners. The material used for those events is available in 
[another GitHub repository](https://github.com/tommasoboccali/ml_infn_hackBase/).

The [third](https://agenda.infn.it/event/32568/) event is targeting advanced users,
and the related material is collected in this repository.

### Structure of the repository
Contents is organized per topic in different folders. 
When documentation beyond the Jupyter notebook is needed, a README.md file is 
included in the sub-directory.

#### Contents
 * `advanced_jupyter`: tricks and suggestions to get the most out of your 
    Jupyter-powered instance including remote access and pipelining multiple notebooks.
 * `introduction_to_gnns`: a gentle introduction to Graph Neural Networks using 
    synthetic data
 * `ex`: material for the hackathon exercises
   * [`unet`](./ex/unet): Lung Segmentation on Chest X-Ray images with U-Net
   * [`domain_adaptation`](./ex/domain_adaptation): Domain Adaptation for model-independent 
     training in High Energy Physics
   * [`gnn_transformers`](./ex/gnn_transformers): Introduction to Graph Neural Networks
     and Transformers with applications from High Energy Physics
   * [`xai](./ex/xai): Introduction to Explainable Artificial Intelligence algorithms
     with applications from Bioinformatics and Biogenetics


### Automated Testing
Tests on the notebooks are run frequently on the different setups being prepared
for the hackathon event.

Run all tests with:
```bash
python3 -m pytest tests/test_notebooks.py -v --durations=0
```

#### Latest results
##### CNAF - RTX 5000 (2022-11-07)
```
1844.80s call     tests/test_notebooks.py::test_xai                                                                                                                                                                           │·····················
772.03s call     tests/test_notebooks.py::test_DA_ML                                                                                                                                                                          │·····················
714.65s call     tests/test_notebooks.py::test_transformers                                                                                                                                                                   │·····················
238.66s call     tests/test_notebooks.py::test_ex_gnn                                                                                                                                                                         │·····················
169.10s call     tests/test_notebooks.py::test_unet_train_only                                                                                                                                                                │·····················
141.26s call     tests/test_notebooks.py::test_intro_gnn                                                                                                                                                                      │·····················
31.29s call     tests/test_notebooks.py::test_intro_pytorch                                                                                                                                                                   │·····················
24.72s call     tests/test_notebooks.py::test_DA_ML_SimpleDNN                                                                                                                                                                 │·····················
17.23s call     tests/test_advanced_jupyter.py::test_snakemake                                                                                                                                                                │·····················
17.03s call     tests/test_notebooks.py::test_unet_arch                                                                                                                                                                       │·····················
12.32s call     tests/test_notebooks.py::test_unet_predict_only                                                                                                                                                               │·····················
3.46s call     tests/test_notebooks.py::test_unet_loss                                                                                                                                                                        │·····················
3.43s call     tests/test_notebooks.py::test_unet_generator                                                                                                                                                                   │·····················
1.17s call     tests/test_notebooks.py::test_unet_intro                                                                                                                                                                       │·····················
```

##### CNAF - A100 with MIG (2022-11-07)
```
1397.29s call     tests/test_notebooks.py::test_xai  
1102.10s call     tests/test_notebooks.py::test_transformers  
556.91s call     tests/test_notebooks.py::test_DA_ML           
186.14s call     tests/test_notebooks.py::test_ex_gnn          
180.75s call     tests/test_notebooks.py::test_unet_train_only 
145.65s call     tests/test_notebooks.py::test_intro_gnn       
28.84s call     tests/test_notebooks.py::test_intro_pytorch    
19.30s call     tests/test_notebooks.py::test_DA_ML_SimpleDNN  
17.04s call     tests/test_notebooks.py::test_unet_arch        
15.57s call     tests/test_advanced_jupyter.py::test_snakemake 
13.05s call     tests/test_notebooks.py::test_unet_predict_only
3.10s call     tests/test_notebooks.py::test_unet_generator    
3.03s call     tests/test_notebooks.py::test_unet_loss         
1.00s call     tests/test_notebooks.py::test_unet_intro
```

#### ReCaS - V100 (2022-11-07)
```
1632.26s call     tests/test_notebooks.py::test_xai
643.56s call     tests/test_notebooks.py::test_DA_ML
609.55s call     tests/test_notebooks.py::test_transformers
318.69s call     tests/test_notebooks.py::test_unet_train_only
224.93s call     tests/test_notebooks.py::test_ex_gnn
182.76s call     tests/test_notebooks.py::test_intro_gnn
48.19s call     tests/test_notebooks.py::test_intro_pytorch
25.11s call     tests/test_notebooks.py::test_DA_ML_SimpleDNN
21.48s call     tests/test_advanced_jupyter.py::test_snakemake
17.07s call     tests/test_notebooks.py::test_unet_predict_only
3.94s call     tests/test_notebooks.py::test_unet_generator
3.66s call     tests/test_notebooks.py::test_unet_loss
1.52s call     tests/test_notebooks.py::test_unet_intro
0.91s call     tests/test_notebooks.py::test_unet_arch
```

# License
Code is released under OSI-approved [MIT license](./LICENSE).

The documentation provided in the form of Jupyter notebooks is 
released under [CC-BY-NC-SA](./CC-BY-NC-SA-4.0) license.

