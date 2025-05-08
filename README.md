# TADC-SBM: a Time-varying, Attributed, Degree-Corrected Stochastic Block Model

This is the code repository for the accompanying paper:

> Passos, N.A.R.A., Carlini, E., Trani, S. (2025). [TADC-SBM: a Time-varying, Attributed, Degree-Corrected Stochastic Block Model](). 2025 IEEE Symposium on Computers and Communications (ISCC), Bologna, Italy, 2025, pp. 1-6.

___

## About

TADC-SBM is a synthetic dataset generator based on [Ghasemian et al. (2016)](http://dx.doi.org/10.1103/PhysRevX.6.031005) and [Tsitsulin et al. (2021)](https://doi.org/10.48550/arXiv.2204.01376) that produces temporal graphs with varying community structures, attribute features, and temporal dynamics, suited for community detection and graph representation learning benchmarks under controlled experimental settings.

## Requires

Requirements can be installed from [pypi (requirements.txt)](requirements.txt) or using [conda (environment.yml)](environment.yml).

## Usage

A command line interface is included to generate graphs:

```
usage: tadcsbm.py ...
```

Please check the referred papers and the example below for details.

### Example

To generate graphs with the same configuration used in the experimental evaluation of the paper:

```
tadcsbm.py --nodes 1024 \
           --snapshots 8 \
           --eta 1 \
           --gamma 0 \
           --alpha 2 \
           --deg-out 2 \
           --deg-min 2 \
           --deg-avg 20 \
           --deg-max 20 \
           --clusters 8 \
           --cluster-var 6 \
           --features 32
```

> Varying the value of $\eta \in [0, 1]$ (`--eta`) produces snapshots with different community stability rates, while the value of $\gamma \in \{0, 1\}$ (`--gamma`) fixes the community transition probabilities for nodes in each snapshot.

The [output](output) files are saved in NetworkX-compatible (GraphML), NumPy, or PyTorch Geometric format.

___

## Cite

```
@inproceedings{tadcsbm2025,
    author = {Reis de Almeida Passos, Nelson Aloysio and Carlini, Emanuele and Trani, Salvatore},
    booktitle={2025 IEEE Symposium on Computers and Communications (ISCC)},
    title={TADC-SBM: a Time-varying, Attributed, Degree-Corrected Stochastic Block Model},
    year={2025},
    volume={},
    number={},
    pages={1-6},
    keywords={Temporal Graphs, Community Detection, Stochastic Block Modeling, Graph Representation Learning},
    doi={XX.XXXX/ISCCYYYYY.YYYY.YYYYYYYY}
}
```