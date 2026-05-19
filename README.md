### Applied Quantum Algorithms Projects
***

# QML Topic 2 - Learning graph invariants with PQCs
Mini-project for the **Applied Quantum Algorithms** course (Leiden, Q3-Q4 2026). 

## Overview

From the [project instructions](aqa_miniprojects_qml_2026.pdf):
>Variational quantum algorithms are candidates for advantage on near-term quantum hardware. The choice of ansatz to solve a specific problem plays an important role in trainibility and performance of the algorithm. So we are constantly looking for more informed ansatzes and the ansatz is used to define the learning model. One way of embedding bias, that is, prior knowledge that will enable your model to learn faster, in your model is through exploiting the symmetries of the problem. In this mini project, you are going to use a particular ansatz, an equivariant ansatz, in a data re-uploading scheme to learn a permutation invariant property of graphs, namely connectivity. A graph is said to be connected if there is a path between every pair of vertices. Then you are going to discuss the expressivity of the quantum model and compare its performance to a classical neural network.

## Repository structure

```
aqal_project/
│
├── papers/                          # Reference literature as provided in the project instructions
│   ├── 2112.05261v3.pdf             # Equivariant Quantum Graph Circuits, Mernyei et al. (2022)
│   └── 2210.07980v2.pdf             # Representation Theory for Geometric Quantum Machine Learning, Ragone et al. (2023)
│
├── plots/                           # Generated figures for the report
│   ├── ...
│
├── quantum_model_variations/        # Variations of the quantum_model notebook for experiments
│   ├── improved_ansatz.ipynb        # Improved ansatz, c.f. section 3.8 of the report
│   ├── non_equiv.ipynb              # Non-equivariant ansatz, c.f. section 3.9 of the report
│   └── unparametrized_M.ipynb       # Non-parametrized data encoding layers, c.f. section 3.4 of the report
│
├── report_utils/                    # Python scripts for data analysis & plots
│   ├── ...
│
├── aqa_miniprojects_qml_2026.pdf    # Project instructions
├── classical_model.ipynb            # Classical k-degree polynomial model
├── experiments_log.jsonl            # Raw experiment logs
├── quantum_model.ipynb              # Main quantum model
├── README.md                        # This file
└── report.zip                       # Packaged back-up of the report (.tex source + figures)
```

## Notebooks

| Notebook | Description                                                                                                          |
|---|----------------------------------------------------------------------------------------------------------------------|
| `quantum_model.ipynb` | Main notebook. Generates graph data, trains the equivariant PQC with the first ansätze, evaluates performance.       |
| `classical_model.ipynb` | Trains the classical multivariate polynomial model $h_k(M)$ and benchmarks it for comparison with the quantum model. |
| `quantum_model_variations/improved_ansatz.ipynb` | Improved ansatz, discussed in section 3.x. of the report.                                                            |
| `quantum_model_variations/non_equiv.ipynb` | Ablation: artificially drops the equivariance property to evaluate its impact.                                       |
| `quantum_model_variations/unparametrized_M.ipynb` | Ablation: disables the data re-uploading layers parametrization (no trainable $\pmb{\gamma}$).                       |

## Methods, results

See [the report](report.pdf).

## Requirements

```
pennylane
numpy
scipy
networkx
matplotlib
scikit-learn
jupyter
```

Install with:

```bash
pip install pennylane numpy scipy networkx matplotlib scikit-learn jupyter
```