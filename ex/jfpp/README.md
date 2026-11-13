# The Added Value of Transformers and Graph Neural Networks for Neuroimaging Analysis

Autism spectrum disorder (ASD) is a heterogeneous neurodevelopmental condition characterized by differences in social communication and interaction, as well as restricted and repetitive patterns of behavior and interests. Neuroimaging studies have identified alterations in both brain morphology and functional connectivity, suggesting that the integration of structural and functional magnetic resonance imaging (MRI) may provide complementary information for ASD classification. The availability of large-scale public datasets, such as the Autism Brain Imaging Data Exchange (ABIDE I and II) [1,2], has facilitated the development and evaluation of machine learning approaches for investigating these differences. Previous studies have demonstrated the potential of multimodal deep learning to combine anatomical and functional information and improve the discrimination of individuals with ASD from typically developing (TD) controls [3].

In this study, we investigate the added value of Transformer-based architectures and graph neural networks (GNNs) within a joint-fusion framework for binary ASD classification [3]. The structural MRI branch employs a Feature Tokenizer Transformer [4] to model interactions among input features, while the functional MRI branch uses a Graph Attention Network (GAT) [5,6] to learn representations of functional brain connectivity. Functional connectivity is represented as a graph, in which brain regions correspond to nodes and their functional relationships to edges. We further investigate the effect of graph sparsification [7] by retaining only the strongest functional connections. 

Beyond predictive accuracy, we examine the explanatory capabilities of the proposed architectures. Specifically, we investigate whether attention-based scores [8] can help identify informative structural features, brain regions, and functional connections, and compare these scores with feature attributions derived from SHapley Additive exPlanations (SHAP) [9]. For the graph-based branch, we also explore whether explanations can be mapped onto the underlying functional connectome to facilitate their neurobiological interpretation. Given ongoing debate about whether attention weights constitute faithful explanations of model predictions [10], we treat attention as a candidate interpretability signal rather than an inherently reliable explanation and assess its agreement with independent attribution methods.

Finally, we evaluate the computational implications of introducing more complex architectures [11]. We compare model complexity, computational requirements, and predictive performance to determine whether potential improvements in classification and interpretability justify the additional computational cost. This analysis follows the broader principles of Green AI, emphasizing the importance of reporting computational resources alongside predictive performance.

## References

1. Di Martino A, Yan CG, Li Q, et al. The autism brain imaging data exchange: towards a large-scale evaluation of the intrinsic brain architecture in autism. *Mol Psychiatry.* 2014;19(6):659-667. doi:10.1038/mp.2013.78

2. Di Martino A, O'Connor D, Chen B, et al. Enhancing studies of the connectome in autism using the autism brain imaging data exchange II. *Sci Data.* 2017;4:170010. doi:10.1038/sdata.2017.10

3. Saponaro S, Lizzi F, Serra G, et al. Deep learning based joint fusion approach to exploit anatomical and functional brain information in autism spectrum disorders. *Brain Inform.* 2024;11(1):2. doi:10.1186/s40708-023-00217-4

4. Gorishniy Y, Rubachev I, Khrulkov V, Babenko A. Revisiting deep learning models for tabular data. In: *Advances in Neural Information Processing Systems.* Vol 34. Curran Associates, Inc; 2021:18932-18943.

5. Veličković P, Cucurull G, Casanova A, Romero A, Liò P, Bengio Y. Graph attention networks. In: *International Conference on Learning Representations.* 2018.

6. Brody S, Alon U, Yahav E. How attentive are graph attention networks? In: *International Conference on Learning Representations.* 2022.

7. Motisi GA, Lizzi F, Mainas F, Tiddia G, Calderoni S, Oliva P, Retico A. A graph neural network approach to study the human connectome in male individuals with autism spectrum disorders. In: *Brain Informatics: 18th International Conference, BI 2025, Bari, Italy, November 11–13, 2025, Proceedings, Part I.* Lecture Notes in Computer Science. Vol 16347. Springer Nature Singapore; 2026:247-258. doi:10.1007/978-981-95-9575-4_19

8. Vaswani A, Shazeer N, Parmar N, et al. Attention is all you need. In: *Advances in Neural Information Processing Systems.* Vol 30. Curran Associates, Inc; 2017.

9. Lundberg SM, Lee SI. A unified approach to interpreting model predictions. In: *Advances in Neural Information Processing Systems.* Vol 30. Curran Associates, Inc; 2017.

10. Jain S, Wallace BC. Attention is not explanation. In: *Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies.* Association for Computational Linguistics; 2019:3543-3556. doi:10.18653/v1/N19-1357

11. Schwartz R, Dodge J, Smith NA, Etzioni O. Green AI. *Commun ACM.* 2020;63(12):54-63. doi:10.1145/3381831

# Exercise
The GitHub directory is divided into two subfolders one with exercise and one for the solution:
```bash
jfpp/
├── ex-solutions
├── ex-students
└── README.md
```
Inside the the exercise subfolder you can find this structure:
```bash
ex-students/
├── data
│   ├── comparison
│   │   ├── computational_cost_per_fold.csv
│   │   ├── jf
│   │   └── jfpp
│   ├── dict.csv
│   ├── functional_features.csv
│   ├── Harmonized_functional_features.csv
│   └── Harmonized_structural_features.csv
├── exercise
│   ├── easy
│   │   ├── 1_Joint_Fusion_easy_ex.ipynb
│   │   └── 2_Joint_Fusion_pp_easy_ex.ipynb
│   ├── medium
│   │   ├── 1_Joint_Fusion_medium_ex.ipynb
│   │   └── 2_Joint_Fusion_pp_medium_ex.ipynb
│   ├── hard
│   └── └── 2_Joint_Fusion_pp_hard_ex.ipynb
├── params
│   ├── best-params_joint-fusion_provided.json
│   ├── jfpp_complete_sgd_l1
│   └── jfpp_knn_strict_sgd_l1
└── weights
    ├── jfpp_complete_sgd_l1
    └── jfpp_knn_strict_sgd_l1
```
At some point of the session you will have access to the solutions. Here you can find the structure:
```bash
ex-solutions/
├── data
│   ├── comparison
│   │   ├── computational_cost_per_fold.csv
│   │   ├── jf
│   │   └── jfpp
│   ├── dict.csv
│   ├── functional_features.csv
│   ├── Harmonized_functional_features.csv
│   └── Harmonized_structural_features.csv
├── Joint_Fusion.ipynb
├── Joint_Fusion_pp.ipynb
├── Comparison.ipynb
├── results
├── figures
├── params
│   ├── jfpp_complete_sgd_l1
│   └── jfpp_knn_strict_sgd_l1
└── weights
    ├── jfpp_complete_sgd_l1
    └── jfpp_knn_strict_sgd_l1
```

## JointFusion notebook

First, you will run the JointFusion model notebook, it is a PyTorch reimplementation of work by Sara Saponaro _et al_ [3].

Each difficulty level includes the tasks of the lower ones. $\checkmark$ = you complete it, $\times$ = already filled in.

| Task | Easy | Medium |
| --- | --- | --- |
| Build the outer stratified 10-fold split | $\checkmark$ | $\checkmark$ |
| Draw the stratified early-stopping subjects from the training set | $\checkmark$ | $\checkmark$ |
| Define the search space of the optimizer: `lr`, `momentum`, `batch_size`, `l1_lambda` | $\checkmark$ | $\checkmark$ |
| Define the search space of the architecture: widths and dropout rates | $\times$ | $\checkmark$ |
| Write the Optuna objective | $\times$ | $\checkmark$ |
| Create the Optuna study with sampler and pruner | $\times$ | $\checkmark$ |
| Run the optimization and save the best parameters | $\checkmark$ | $\checkmark$ |
| Train and evaluate the joint model with the tuned hyperparameters | $\times$ | $\checkmark$ |

## JointFusion++ notebook

In this notebook we introduce the Joint Fusion++ (JF++) model. It repeats the analysis of `Joint_Fusion.ipynb` (structural-only model, functional-only model, joint model, 10-fold cross-validation, hyperparameter search with Optuna) but replaces the two MLP encoders with graph / attention encoders:

```
structural (221)  ── FT-Transformer (feature tokens + [CLS]) ── embedding ──┐
                                                                            ├─ concat ── fusion head ── logit
functional (5253) ── per-subject brain graph (103 regions) ── GATv2 ─────────┘
```

The 5253 functional features are the upper triangle of the $103 \times 103$ functional-connectivity matrix, so each subject is a **graph**: 103 nodes (brain regions), one candidate edge per pair of regions. The graph can be **complete** (all 5253 connections) or **sparsified** with a per-subject **kNN rule**. In this notebook we will use:
1. **kNN strict** (with `k = 1`): only the connections kept by the kNN rule are edges, *and* the removed connections are set to 0 in the node features;
2. **complete**: the reference without sparsification.

Each difficulty level includes the tasks of the lower ones. $\checkmark$ = you complete it, $\times$ = already filled in.

| Task | Easy | Medium | Hard |
| --- | --- | --- | --- |
| `N_FUNC` as a function of `N_REGIONS` | $\checkmark$ | $\checkmark$ | $\checkmark$ |
| `load_data`: merge the structural and functional tables | $\checkmark$ | $\checkmark$ | $\checkmark$ |
| `load_data`: convert features to `np.float64` | $\checkmark$ | $\checkmark$ | $\checkmark$ |
| `fit_scaler`: `RobustScaler` fitted on the training rows only | $\checkmark$ | $\checkmark$ | $\checkmark$ |
| Build `OUTER` folds and print the fold sizes | $\checkmark$ | $\checkmark$ | $\checkmark$ |
| `FunctionalGAT`: `BatchNorm1d` input size | $\checkmark$ | $\checkmark$ | $\checkmark$ |
| `MorphologicalTransformer`: tokenizer, CLS token and head | $\checkmark$ | $\checkmark$ | $\checkmark$ |
| `make_optimizer` (SGD + momentum) | $\checkmark$ | $\checkmark$ | $\checkmark$ |
| Data-leakage check on fold 0 | $\times$ | $\checkmark$ | $\checkmark$ |
| `knn_keep` (kNN sparsification) | $\times$ | $\checkmark$ | $\checkmark$ |
| `FunctionalGAT`: `gat1` and `gat2` (`GATv2Conv`) | $\times$ | $\checkmark$ | $\checkmark$ |
| `AttnBlock`: `nn.MultiheadAttention` | $\times$ | $\checkmark$ | $\checkmark$ |
| `JointFusionModel`: `self.gat` and `self.morph` | $\times$ | $\checkmark$ | $\checkmark$ |
| `MorphOnlyModel`: `self.morph` | $\times$ | $\checkmark$ | $\checkmark$ |
| `FuncOnlyModel`: `self.gat` | $\times$ | $\checkmark$ | $\checkmark$ |
| `predict_proba` | $\times$ | $\checkmark$ | $\checkmark$ |
| `FunctionalGAT.forward` | $\times$ | $\times$ | $\checkmark$ |
| `AttnBlock.forward` (pre-norm transformer block) | $\times$ | $\times$ | $\checkmark$ |

## Comparison notebook

This notebook compares the original **Joint Fusion (JF**, MLP encoders, Saponaro et al. 2024) with **Joint Fusion++ (JF++**, GATv2 + FT-Transformer). Both models are trained with the same protocol (10 outer folds `StratifiedKFold(10, shuffle, 13)`, SGD + L1, nested Optuna search), so every comparison is **paired by fold**.

Results, previously computed, are stored in `./data/comparison/` (copied from the project repository):

| Model | Folder in `./data/comparison/` (original run folder) |
|---|---|
| JF | `jf/` |
| JF++ (kNN strict) | `jfpp/`|
|Computational cost | `computational_cost_per_fold.csv` |

At the end of the notebook, we will try to answear together to these three questions:

1. **Does the new model improve classification performance?** (Section 1)
2. **What happens to the explanations of the two models, and how do they change between attention and SHAP?** (Section 2)
3. **What is the computational cost of the new model compared to the current one?** (Section 3)

No tasks in this notebook, but we prepare some cells where you can submit your answers.