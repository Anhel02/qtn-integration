# Quantum Tensor Networks Integration

This repository contains Python code developed for a Master's Thesis in Advanced Physics on the Universidad de Valencia, focused on solving multidimensional Feynman integrals using quantum computing techniques. The program makes use of Singular Value Decomposition (SVD) and Tensor Train (TT) decomposition for encoding a non-uniform sampling distribution of an integrand based on Importance Sampling (IS) into a quantum circuit for its subsequent estimation.

## Overview

This repository implements a hybrid quantum-classical pipeline for evaluating multidimensional Feynman integrals. The workflow maps a non-uniform sampling distribution into a quantum circuit, allowing us to estimate the integral's solution through quantum measurements.

- **Discretization & Compression (TT-SVD):** The integration space is discretized, and the probability distribution is processed using Oseledets' TT-SVD algorithm. This transforms the data into a TT format, allowing us to apply truncation to reduce the complexity of the state.
- **Circuit Generation (TT-QC):** The compressed TT format is then processed via the TT-QC algorithm proposed by Pereira et al. This step translates the tensors into unitary matrices that can be directly mapped to a quantum circuit.
- **Quantum Estimation:** The circuit is executed for multiple shots. The measured probability distribution approximates the original integrand, yielding the integral solution.

The repository includes general 1D and 2D integration codes, alongside implementations for evaluating the Tadpole, Bubble, and Triangle Feynman diagrams in the Loop-Tree Duality (LTD) framework. Additionally, a Mathematica notebook is provided with a step-by-step example of the entire pipeline.

## Requirements

To run the program pennylane and jax are needed while matplotlib is only needed for plotting the circuit. They can be installed by running
```bash
pip install -r requirements.txt
```

## How to Run

### 1D Case

The 1D integration case can be run by simply executing the python script from the terminal:
```bash
python 1d.py
```
As an example, a $\sin^2(\tfrac{1}{x})$ function is setted, but it can be changed to any 1d function in the `integrand()` definition.

Before running, it can be easily configured the algorithm's hyperparameters directly in the file:
- `n`: Number of qubits that defines the discretization resolution.
- `rel_acc`: Relative accuracy for the TT-SVD truncation (setting it to 0 means no truncation).
- `shots`: Number of quantum measurements.
- `x_min` and `x_max`: Integration domain limits.

Additionaly, there can be also toggled optional flags:
- `transpile`: Compiles the circuit into `CZ` and `U3` universal basis gates (required in real hardware).
- `show_qc`: Displays a plot of the quantum circuit.
- `show_specs`: Prints PennyLane's circuit specifications.

Run the script to compute the estimated energy values for different spatial resolutions (determined by d) and evolution steps:

### 2D Case

The 2D case can also be executed by running:
```bash
python 2d.py
```
Here, the default function is $\sin^2(\tfrac{1}{x}+\tfrac{1}{y})$, but it can also be changed in the `integrand()` definition.

Furthermore, the same hyperparameters as in the 1D case can be changed. However, the 2D case also allows chaning the mapping strategy used to flatten the multi-dimensional probability matrix into a probability vector. By default, it is set to `interleaving` which is the optimal for capturing correlations between the variables, but it can be set to `sequential` or `mirroring`.

### LTD Feynman Diagrams

The `LTD` subfolder contains the specific parametrizations for the three Feynman integrals. These scripts can be executed in the exact same manner as the generic cases described above, since the Tadpole and Bubble diagrams are evaluated as 1D cases, while the Triangle diagram is a 2D case.

## References

This implementation is based on the mathematical and computational frameworks proposed in the following papers:

Oseledets, I. V. (2011).
*Tensor-Train Decomposition.*
SIAM Journal on Scientific Computing, 33(5), 2295-2317.  
[DOI: 10.1137/090752286](https://doi.org/10.1137/090752286)

Pereira, A., Villarino, A., Cortines, A., Mugel, S., Orús, R., Leme Beltran, V., Scursulim, J. V. S., & Brito, S. (2024).
*Encoding of Probability Distributions for Quantum Monte Carlo Using Tensor Networks.*
arXiv preprint arXiv:2411.11660.  
[DOI: 10.48550/arXiv.2411.11660](https://doi.org/10.48550/arXiv.2411.11660)

## Acknowledgments

Special thanks to Dr. Germán Rodrigo for his guidance throughout the project.
