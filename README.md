# Operational Regimes in Non-Convex Optimization: A Multiplier-Based Taxonomy

> **Local review/refactoring snapshot — not an upstream release.** See
> [REFACTORING.md](REFACTORING.md) for the new CLI, validation scope and limitations.
> The original README below describes the upstream project, not a certification
> that this modified snapshot reproduces the paper.

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

At a KKT point of a constrained nonlinear program, Lagrange multipliers encode sensitivity to constraint levels. This paper studies their **collective geometry**: the total magnitude of the active inequality reactions, how that reaction is distributed across constraints, and how the active constraint normals interact directionally.

The paper defines a three-component multiplier signature
\[
\Xi=(\Lambda,\vartheta,I),
\]
constructed from the KKT multipliers and active constraint gradients. It is used to assign a KKT point to one of **four open core operational regimes**, with a **hybrid interface** comprising the threshold boundaries.

### Multiplier signature

Let \(x^\star\) be a KKT point, and let \(J=\operatorname{supp}(\lambda^\star)\) be the strongly active inequality constraints (those with positive multipliers). For \(i\in J\), define the gradient-normalized reaction
\[
\widetilde{\lambda}_i
=\lambda_i^\star\|\nabla g_i(x^\star)\|_2,
\qquad
\Lambda=\sum_{i\in J}\widetilde{\lambda}_i.
\]
When \(\Lambda>0\), normalize the reaction profile as
\[
\bar{\lambda}_i=\frac{\widetilde{\lambda}_i}{\Lambda}.
\]

The signature comprises:

- **Sensitivity mass \(\Lambda\):** the total gradient-normalized reaction of the strongly active inequality constraints. It carries units and is compared with a reference-scaled threshold \(\delta_\Lambda\).
- **Effective participation ratio \(\vartheta\):**
  \[
  \vartheta(\bar{\lambda})
  =\frac{1}{\|\bar{\lambda}\|_2^2}
  =\frac{\Lambda^2}{\|\widetilde{\lambda}\|_2^2}.
  \]
  This is an effective count of how many constraints share the reaction; it is the inverse concentration (Herfindahl–Hirschman/Simpson) index of the normalized reaction profile.
- **Support-normalized interference index \(I\):** measures the size of the individual reactions relative to their net vector force,
  \[
  I(x^\star,\lambda^\star)
  =\frac{\|\widetilde{\lambda}\|_2}
  {\left\|\sum_{i\in J}\lambda_i^\star\nabla g_i(x^\star)\right\|_2}.
  \]
  Equivalently, it is determined by the normalized active-normal Gram matrix. \(I=1\) for mutually orthogonal active normals; \(I<1\) indicates weighted constructive alignment, while \(I>1\) indicates weighted destructive interference/cancellation.

If \(\Lambda=0\), the paper adopts a convention for the shape coordinates and labels the point `unc`; in the presence of equality constraints, this means **no active inequality reaction**, not necessarily a zero objective gradient.

### Operational regimes

The classification is defined using thresholds
\[
\Theta=(\delta_\Lambda,\kappa_{\rm srd},
\kappa_{\rm sat},\gamma_I),
\]
subject to the paper's admissibility conditions. In particular,
\(1<\kappa_{\rm srd}<2<\kappa_{\rm sat}<n-p\),
\(\kappa_{\rm sat}\notin\mathbb{N}\), \(\gamma_I>1\), and
\(\delta_\Lambda>0\). The definitions below apply to the strongly active dual face and use strict inequalities; threshold-boundary cases belong to the hybrid interface.

| Label | Regime name in the paper | Defining conditions | Interpretation |
| --- | --- | --- | --- |
| `unc` | Negligible constraint reaction | \(\Lambda<\delta_\Lambda\) | Active inequality reaction is below the prescribed mass threshold. |
| `srd` | Single-resource dominant | \(\Lambda>\delta_\Lambda,\ \vartheta<\kappa_{\rm srd}\) | The reaction is concentrated: the largest normalized reaction share is guaranteed to exceed \(1/\kappa_{\rm srd}>1/2\). |
| `int` | Intermediate / interference-coupled | \(\Lambda>\delta_\Lambda,\ \vartheta>\kappa_{\rm srd}\), and \((\vartheta<\kappa_{\rm sat}\ \text{or}\ I>\gamma_I)\) | Either participation is intermediate, or sufficiently strong destructive interference places the point in this regime. |
| `sat` | Saturated / distributed | \(\Lambda>\delta_\Lambda,\ \vartheta>\kappa_{\rm sat},\ I<\gamma_I\) | Reaction is broadly distributed and does not exceed the interference threshold. |
| `hyb` | Hybrid interface | Threshold-boundary cases separating the open core regimes | A transition/interface label, not a fifth open core regime. |

For a given active set, `sat` may be unavailable when the effective participation ratio cannot exceed \(\kappa_{\rm sat}\); in particular, small active sets may realize only `unc`, `srd`, and `int`. Threshold calibration is consequential, and the Euclidean gradient normalization presumes that the decision variables have compatible units or have been appropriately normalized.

### Scope and theoretical results

Under the regularity assumptions stated in the paper (including LICQ and second-order sufficiency, with strict complementarity where required), the analysis establishes properties of this KKT-point diagnostic, including:

- invariance under constraint permutations and positive constraint/objective rescalings, with the mass threshold transformed covariantly; invariance under the specified Euclidean similarities and locally isometric coordinate changes;
- local constancy of the regime label under sufficiently small perturbations, with an explicit stability radius;
- a transition-set and boundary analysis for the hybrid interface, including transversal parameter paths;
- an exact linear-time classifier at zero tolerance, together with soundness/accuracy bounds for inexact solver iterates, sampled data, and drifting data.

The taxonomy is intended as an inexpensive, solver-independent diagnostic of the dual geometry at a KKT point—not as a demonstrated solver-acceleration method. The paper's regime-related solver-behaviour claims are hypotheses except for the stated local spectral result; its regime-aware ADMM experiment did **not** outperform residual balancing.

## Key contributions

- **A three-scalar KKT signature:** gradient-normalized reaction mass \(\Lambda\), effective participation ratio \(\vartheta\), and support-normalized interference index \(I\).
- **Four open core regimes plus a hybrid boundary:** `unc`, `srd`, `int`, and `sat`, with definitions governed by the threshold tuple \(\Theta\); `hyb` denotes threshold-interface cases.
- **Structural analysis:** invariance, local stability, regime-transition geometry, and the topology of the hybrid interface under the paper's assumptions.
- **Classifier and finite-error guarantees:** an \(O(m+|\widehat A|n)\) classification procedure, with results for solver-iterate stabilization, sampling uncertainty, and online tracking under bounded drift.
- **Numerical study:** controlled synthetic families, Maros–Mészáros quadratic-program benchmarks, and multi-user MIMO downlink beamforming. The experiments support the mathematical analysis while also documenting conservatism of the bounds, regularity limitations, threshold-calibration sensitivity, and the lack of improvement of the tested regime-aware ADMM rule over residual balancing.

For exact theorem statements, assumptions, threshold conventions, and experimental protocols, consult the paper: [arXiv:2609.00471](https://arxiv.org/abs/2609.00471).

## Repository structure

The following is a suggested structure for organizing code and artifacts associated with the paper. Adjust it to match the files actually committed to this repository.

```text
.
├── README.md
├── LICENSE                  # GPL-3.0 license for source code
├── CITATION.cff             # Optional citation metadata
├── requirements.txt         # Python dependencies, if applicable
├── pyproject.toml
├── configs/
├── data/                    # Benchmark instances and derived data
├── src/
│   └── operational_regimes/
│       ├── __init__.py
│       ├── __main__.py
│       ├── cli.py
│       ├── regimes/         # Regime definitions and classification
│       ├── optimization/    # Optimization and solver interfaces
│       └── utils/           # Shared utilities
├── experiments/             # Experiment and reproduction scripts
│   ├── synthetic/
│   ├── drift/
│   ├── benchmarks/
│   ├── mimo/
│   ├── admm/
│   └── figures/
├── tests/                   # Automated tests
└── results/
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
