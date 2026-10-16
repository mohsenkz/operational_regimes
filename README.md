# Operational Regimes in Non-Convex Optimization: A Multiplier-Based Taxonomy

[![arXiv](https://img.shields.io/badge/arXiv-2609.00471-b31b1b.svg)](https://arxiv.org/abs/2609.00471)
[![Code License: GPL v3](https://img.shields.io/badge/Code-GPL--3.0-blue.svg)](LICENSE)
[![Documentation License: CC BY 4.0](https://img.shields.io/badge/Documentation-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

Official repository for the paper:

**Operational Regimes in Non-Convex Optimization: A Multiplier-Based Taxonomy**

**Authors:** Seyed Mohsen Kazemi, Ali Movaghar, and Shaahin Hessabi  
**arXiv:** [2609.00471](https://arxiv.org/abs/2609.00471)  
**DOI:** [10.48550/arXiv.2609.00471](https://doi.org/10.48550/arXiv.2609.00471)

---

## Overview

Constrained non-convex optimization appears in many areas, including signal processing, wireless communications, and machine learning. Although a wide range of algorithms are available, selecting an appropriate method and understanding how a problem responds to changes in its data remain challenging.

This work introduces a **multiplier-based taxonomy of operational regimes** for constrained non-convex optimization. The central idea is that the normalized vector of Lagrange multipliers at a KKT stationary point can provide an algorithm-independent structural signature of the local optimization problem.

The framework uses four scale-free shape features of the normalized multiplier vector to distinguish five operational regimes:

1. **Unconstrained**
2. **Resource-Limited**
3. **Saturation**
4. **Strongly-Coupled**
5. **Hybrid**

The paper develops structural results for this partition and proposes a linear-time classifier for identifying regimes. Its theoretical analysis addresses invariance under natural KKT symmetries, local stability under data perturbations, regime transitions, and the topological role of the Hybrid regime. Numerical evaluations include 104 mixed-integer nonlinear programming (MINLP) instances and a downlink beamforming instance. See the paper for the precise definitions, assumptions, and results.

## Key contributions

- **Multiplier-based structural taxonomy:** Uses normalized Lagrange multiplier information to characterize operational regimes at KKT stationary points.
- **Five-regime partition:** Defines Unconstrained, Resource-Limited, Saturation, Strongly-Coupled, and Hybrid regimes using four scale-free shape features.
- **Algorithm-independent perspective:** Relates the taxonomy to a unified game-theoretic interpretation of eight algorithm families: block coordinate descent (BCD), alternating direction method of multipliers (ADMM), generalized Benders decomposition (GBD), successive convex approximation (SCA), interior-point methods, mirror descent, Frank–Wolfe, and Riemannian gradient descent.
- **Structural guarantees:** Studies invariance, perturbation stability using Robinson's strong regularity, codimension-one regime transitions, and the topology of the Hybrid regime.
- **Linear-time regime classification:** Proposes a classifier and analyzes its correctness, stabilization, sample complexity, and online tracking under data drift.
- **Numerical evaluation:** Reports experiments on 104 MINLP instances and a downlink beamforming problem.

For formal statements and their conditions, consult the full paper rather than relying on this summary.

## Repository structure

The following is a suggested structure for organizing code and artifacts associated with the paper. Adjust it to match the files actually committed to this repository.

```text
.
├── README.md
├── LICENSE                  # GPL-3.0 license for source code
├── CITATION.cff             # Optional citation metadata
├── requirements.txt         # Python dependencies, if applicable
├── data/                    # Benchmark instances and derived data
├── src/
│   ├── regimes/             # Regime definitions and classification
│   ├── optimization/        # Optimization and solver interfaces
│   └── utils/               # Shared utilities
├── experiments/             # Experiment and reproduction scripts
└── tests/                   # Automated tests
```

> **Note:** The tree above is a documentation template, not a guarantee that every listed directory or script is currently present. Keep this section synchronized with the actual repository contents.

## Installation

### Requirements

Check the implementation and dependency files in this repository for the supported environment. If the project uses Python and includes `requirements.txt`, the following setup is a typical starting point.

```bash
git clone https://github.com/mohsenkz/operationalregimes.git
cd operationalregimes

python -m venv .venv

# Linux/macOS
source .venv/bin/activate

# Windows PowerShell
# .venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt
```

If `requirements.txt` or the referenced source directories are not present, do not run these commands as-is; follow the installation instructions provided with the actual implementation.

## Usage

The intended workflow is:

1. Define or load a constrained non-convex optimization instance.
2. Obtain a KKT stationary point and the corresponding Lagrange multiplier information, using the assumptions and conventions specified in the paper.
3. Normalize the multiplier vector and compute the four shape features.
4. Apply the regime-classification procedure.
5. Evaluate classification behavior and robustness using the relevant benchmark or experiment scripts.

The exact API, command-line entry points, solver requirements, and configuration options depend on the code released in this repository. No unverified function names or experiment commands are specified here. Once the implementation is committed, add a minimal executable example and the exact commands needed to reproduce the paper's reported experiments.

## Reproducing the experiments

The paper reports numerical evaluations on 104 MINLP instances and a downlink beamforming instance. To make experimental results reproducible, document the following for each experiment:

- benchmark source and instance identifiers;
- solver names and versions;
- stopping criteria and numerical tolerances;
- initialization strategy and random seeds, where applicable;
- the method used to obtain KKT points and multipliers;
- normalization conventions and regime-classifier parameters;
- software versions, hardware details when relevant, and exact run commands.

Keep a clear distinction between results reported in the paper, results reproduced from the repository, and results from new experiments.

**Paper and supplementary details:** [https://arxiv.org/abs/2609.00471](https://arxiv.org/abs/2609.00471)

## Citation

If you use this work in academic research, please cite the paper:

```bibtex
@article{kazemi2026operational,
  title         = {Operational Regimes in Non-Convex Optimization: A Multiplier-Based Taxonomy},
  author        = {Kazemi, Seyed Mohsen and Movaghar, Ali and Hessabi, Shaahin},
  journal       = {arXiv preprint arXiv:2609.00471},
  year          = {2026},
  eprint        = {2609.00471},
  archivePrefix = {arXiv},
  primaryClass  = {math.OC},
  doi           = {10.48550/arXiv.2609.00471},
  url           = {https://arxiv.org/abs/2609.00471}
}
```

## License and rights

This repository follows the intended dual-licensing arrangement:

- **Source code — GNU General Public License v3.0 (GPL-3.0):** Source code is licensed under GPL-3.0. See the repository's `LICENSE` file for the license text and terms.
- **Documentation and other designated textual materials — Creative Commons Attribution 4.0 International (CC BY 4.0):** These materials may be shared and adapted under the terms of CC BY 4.0, provided appropriate attribution is given. See the [CC BY 4.0 legal code](https://creativecommons.org/licenses/by/4.0/).

The applicable license depends on the type of material. Do not assume that third-party datasets, dependencies, or external assets are covered by either license; their own license terms may apply. Ensure that the repository includes the relevant license files and that each license is applied to the intended materials.

## Contributing

Contributions that improve the implementation, tests, documentation, or reproducibility are welcome. When reporting a bug or suggesting a change, include:

- a concise description of the issue;
- steps to reproduce it;
- expected and observed behavior;
- relevant instance/configuration details and software versions.

Please keep proposed changes consistent with the mathematical definitions and assumptions in the paper.

---

**Paper:** [Operational Regimes in Non-Convex Optimization: A Multiplier-Based Taxonomy](https://arxiv.org/abs/2609.00471)
