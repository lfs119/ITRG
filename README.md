# ITRG

**ITRG** is an interface-centered inference-time extension for Boltz-2 that supports controllable ensemble generation of biomolecular complexes. It combines contact-aware pair-feature perturbation, MCMC-based interface path construction, and descriptor-specific repulsion-guided sampling to explore alternative receptor-partner interface states.

This repository is intended for users who want to generate and analyze alternative interface conformations for protein-small-molecule, protein-peptide, and protein-nucleic-acid complexes using Boltz-2 as the underlying structure generation engine.

> **Note**  
> ITRG is an independent research implementation built on the publicly available Boltz-2 framework. It is not an official release of the Boltz-2 developers.

---

## Key Features

ITRG extends standard Boltz-2 inference with three complementary modules.

### 1. Contact-aware beta interface perturbation

ITRG extracts Boltz-2 pair features and selectively perturbs only receptor-partner pair-feature entries associated with the interface contact matrix. The perturbation strength is controlled by the `--beta_z_interface` parameter.

This module is designed to test how generated complex ensembles respond to controlled modulation of interface-associated pair features.

Typical use cases include:

- Broadening protein-ligand contact-mode landscapes.
- Comparing moderate and stronger perturbation regimes.
- Evaluating residue-level contact redistribution relative to standard Boltz-2 generation.

### 2. MCMC-based interface path construction

ITRG can sample a sequence of perturbed interface feature states using contact-guided MCMC. Each retained feature state is used to condition Boltz-2 generation, producing a local ensemble. Representative structures are then ordered according to the MCMC sequence to form a contact-resolved interface trajectory.

This module is designed for analyzing:

- Stepwise interface remodeling.
- Anchor-referenced contact retention or loss.
- Contact overlap and patch overlap along a generated trajectory.

### 3. Descriptor-specific repulsion-guided sampling

ITRG introduces batch-level repulsive guidance during Boltz-2 diffusion generation. The guidance can act in three descriptor spaces:

- `conformer`: diversifies ligand internal-distance geometry.
- `pose`: diversifies ligand placement after protein alignment.
- `environment`: diversifies the dynamic-shell local binding environment around the ligand.

This module is designed to reduce near-duplicate samples and expand selected diversity modes during generation.

---

## Repository Contents

A typical repository layout is expected to include:

```text
ITRG/
├── README.md
├── run_guidance.sh
├── run_pocket_interface.sh
├── examples/
│   ├── T4L/
│   │   └── 4W53.yaml
│   └── affinity_potentials_debug.yaml
├── docs/
│   └── prediction.md
└── boltz/
    └── ...
```

The two provided shell scripts illustrate the main workflows:

- `run_guidance.sh`: baseline Boltz-2 generation and descriptor-specific repulsion-guided sampling.
- `run_pocket_interface.sh`: beta interface steering and MCMC path generation.

Paths in the example scripts may point to local cluster directories. Before running, update YAML input paths, prior complex paths, GPU IDs, and output locations according to your environment.

---

## Installation

Clone the repository and install it in editable mode:

```bash
git clone https://github.com/lfs119/ITRG.git
cd ITRG
pip install -e .[cuda]
```

For CPU-only or non-CUDA environments, remove `[cuda]`:

```bash
pip install -e .
```

GPU inference is strongly recommended. CPU inference can be substantially slower, especially when using a large number of diffusion samples or MCMC states.

---

## Basic Boltz-2 Inference

Standard Boltz-2 inference can be run with:

```bash
boltz predict input.yaml --use_msa_server
```

where `input.yaml` is a Boltz-compatible YAML file describing the receptor, partner molecule or chain, and optional prediction settings.

You can also provide a directory containing multiple YAML files for batched prediction:

```bash
boltz predict input_directory/ --use_msa_server
```

To view all available options:

```bash
boltz predict --help
```

---

## Input Files

ITRG uses the same general YAML input format as Boltz-2. The input should specify the biomolecular system to be modeled, such as:

- A protein-small-molecule complex.
- A protein-peptide complex.
- A protein-DNA or protein-RNA complex.

For interface steering and repulsion-guided sampling, a prior or reference complex structure is often needed to define the receptor-partner interface, contact mask, or protein patch:

```bash
--prior_complex_path path/to/reference_complex.cif
```

This prior complex is used to identify the relevant interface region. It should correspond to the same receptor-partner system as the YAML input.

---

## Workflow 1: Descriptor-Specific Repulsion-Guided Sampling

The script `run_guidance.sh` demonstrates four runs for the 4W53 protein-ligand system:

1. Standard Boltz-2 baseline generation.
2. Environment-guided diversity sampling.
3. Pose-guided diversity sampling.
4. Conformer-guided diversity sampling.

### Example baseline generation

```bash
CUDA_VISIBLE_DEVICES=0 boltz predict examples/T4L/4W53.yaml \
  --use_msa_server \
  --use_potentials \
  --recycling_steps 10 \
  --sampling_steps 200 \
  --diffusion_samples 100 \
  --override \
  --seed 42
```

### Example environment-guided sampling

```bash
CUDA_VISIBLE_DEVICES=0 boltz predict examples/T4L/4W53.yaml \
  --enable_protein_patch_mask \
  --use_msa_server \
  --use_potentials \
  --recycling_steps 10 \
  --sampling_steps 200 \
  --diffusion_samples 100 \
  --override \
  --ligand_diversity_mode environment \
  --flag env \
  --seed 42 \
  --prior_complex_path path/to/prior_complex.cif
```

### Example pose-guided sampling

```bash
CUDA_VISIBLE_DEVICES=0 boltz predict examples/T4L/4W53.yaml \
  --enable_protein_patch_mask \
  --use_msa_server \
  --use_potentials \
  --recycling_steps 10 \
  --sampling_steps 200 \
  --diffusion_samples 100 \
  --override \
  --ligand_diversity_mode pose \
  --flag pose \
  --seed 42 \
  --prior_complex_path path/to/prior_complex.cif
```

### Example conformer-guided sampling

```bash
CUDA_VISIBLE_DEVICES=0 boltz predict examples/T4L/4W53.yaml \
  --enable_protein_patch_mask \
  --use_msa_server \
  --use_potentials \
  --recycling_steps 10 \
  --sampling_steps 200 \
  --diffusion_samples 100 \
  --override \
  --ligand_diversity_mode conformer \
  --flag conformer \
  --seed 42 \
  --prior_complex_path path/to/prior_complex.cif
```

### Important options

| Option | Description |
|---|---|
| `--ligand_diversity_mode conformer` | Expands ligand internal-distance diversity. |
| `--ligand_diversity_mode pose` | Expands protein-aligned ligand placement diversity. |
| `--ligand_diversity_mode environment` | Expands local binding-environment diversity. |
| `--enable_protein_patch_mask` | Enables patch/interface masking based on the prior complex. |
| `--prior_complex_path` | Path to the prior complex used to define the interface or local environment. |
| `--diffusion_samples` | Number of samples generated in one prediction run. |
| `--sampling_steps` | Number of diffusion sampling steps. |
| `--recycling_steps` | Number of Boltz-2 recycling steps. |
| `--flag` | Output label used to distinguish baseline, conformer, pose, or environment runs. |

---

## Workflow 2: Beta Interface Steering and MCMC Path Generation

The script `run_pocket_interface.sh` demonstrates beta interface steering and MCMC trajectory generation. It includes four example settings:

1. Anchor generation with `beta = -1`.
2. Anchor generation with `beta = -2`.
3. MCMC path generation with `beta = -2`.
4. MCMC path generation with `beta = -1`.

### Example anchor run with beta = -1

```bash
CUDA_VISIBLE_DEVICES=0 boltz predict examples/affinity_potentials_debug.yaml \
  --enable_protein_patch_mask \
  --use_msa_server \
  --use_potentials \
  --recycling_steps 10 \
  --sampling_steps 200 \
  --diffusion_samples 100 \
  --override \
  --enable_interface_steering \
  --beta_z_interface -1 \
  --max_mcmc_samples 1 \
  --seed 42 \
  --flag f1 \
  --protein_chain_names A \
  --partner_chain_names B \
  --interface_chain B \
  --prior_complex_path path/to/prior_complex.cif
```

### Example anchor run with beta = -2

```bash
CUDA_VISIBLE_DEVICES=0 boltz predict examples/affinity_potentials_debug.yaml \
  --enable_protein_patch_mask \
  --use_msa_server \
  --use_potentials \
  --recycling_steps 10 \
  --sampling_steps 200 \
  --diffusion_samples 100 \
  --override \
  --enable_interface_steering \
  --beta_z_interface -2 \
  --max_mcmc_samples 1 \
  --seed 42 \
  --flag f2 \
  --protein_chain_names A \
  --partner_chain_names B \
  --interface_chain B \
  --prior_complex_path path/to/prior_complex.cif
```

### Example MCMC run with beta = -2

```bash
CUDA_VISIBLE_DEVICES=0 boltz predict examples/affinity_potentials_debug.yaml \
  --enable_protein_patch_mask \
  --use_msa_server \
  --use_potentials \
  --recycling_steps 10 \
  --sampling_steps 200 \
  --diffusion_samples 100 \
  --override \
  --enable_interface_steering \
  --beta_z_interface -2 \
  --max_mcmc_samples 10 \
  --seed 42 \
  --flag MCMCf2 \
  --protein_chain_names A \
  --partner_chain_names B \
  --interface_chain B \
  --prior_complex_path path/to/prior_complex.cif
```

### Important options

| Option | Description |
|---|---|
| `--enable_interface_steering` | Enables beta-parameterized interface pair-feature perturbation. |
| `--beta_z_interface` | Dimensionless steering coefficient for contact-associated pair features. Typical manuscript settings are `-1` and `-2`. |
| `--max_mcmc_samples 1` | Generates only the beta-perturbed anchor state. |
| `--max_mcmc_samples 10` | Generates a sequence of MCMC-sampled interface feature states. |
| `--protein_chain_names` | Receptor protein chain names used for interface definition. |
| `--partner_chain_names` | Partner chain names used for contact analysis. |
| `--interface_chain` | Chain used as the partner/interface chain for masking. |
| `--prior_complex_path` | Prior complex structure used to derive the contact mask. |
| `--flag` | Output label for distinguishing beta and MCMC settings. |

---

## Parameter Notes

### Beta steering coefficient

The beta value controls the magnitude and direction of contact-aware pair-feature modulation. In the manuscript, `beta = -1` and `beta = -2` are used as representative perturbation regimes:

- `beta = -1`: moderate perturbation.
- `beta = -2`: stronger perturbation.

These values are not intended to be universally optimal. For a new system, beta should be treated as a tunable inference-time hyperparameter.

### Number of diffusion samples

The example scripts use:

```bash
--diffusion_samples 100
```

This setting generates 100 samples per run or per MCMC state. Increasing this number improves ensemble statistics but increases GPU memory and runtime.

### Sampling and recycling steps

The example scripts use:

```bash
--sampling_steps 200
--recycling_steps 10
```

These settings follow the experimental configuration used for the manuscript examples. Users may adjust them depending on speed, accuracy, and hardware constraints.

### Prior complex path

The prior complex should be a structurally compatible receptor-partner complex, usually a `.cif` file generated by Boltz-2 or an experimentally resolved complex structure. It is used to define the interface mask, protein patch, and local binding environment.

---

## Expected Outputs

The `boltz predict` command generates output directories containing predicted structures and associated result files. The exact directory name depends on the input YAML name and the `--flag` argument.

Common outputs include:

- Predicted complex structures in CIF format.
- Multiple generated samples for each input system.
- Flag-specific result directories for baseline, beta-steered, MCMC, conformer-guided, pose-guided, and environment-guided runs.

For manuscript-level analysis, generated structures can be further processed to compute:

- Contact probability maps.
- Delta contact probability maps.
- Contact-mode clusters.
- Contact overlap and patch overlap.
- Ligand conformer diversity.
- Protein-aligned ligand pose diversity.
- Binding-environment descriptor diversity.
- Minimum interface distances and clash statistics.

---

## Reproducing the Main Manuscript Workflows

The manuscript uses five representative receptor-partner systems:

| System | Interface type | Role in analysis |
|---|---|---|
| 7UX8 | MfnG O-methyltransferase / L-tyrosine | Primary contact-rich polar small-molecule case. |
| 4W53 | T4 lysozyme L99A / toluene | Compact hydrophobic buried-cavity small-molecule case. |
| 1TGH | Human TATA-binding protein / TATA-sequence DNA | Protein-DNA extension with base/backbone contact channels. |
| 2ERR | Fox-1 RNA-binding domain / UGCAUGU RNA | Protein-RNA extension with base/backbone contact channels. |
| 1BBZ | Abl-SH3 domain / high-affinity peptide ligand | Protein-peptide extension using total-interface analysis. |

For protein-small-molecule systems, run:

1. Standard Boltz-2 baseline generation.
2. Beta-steered generation with `--beta_z_interface -1` and `--beta_z_interface -2`.
3. MCMC path generation with `--max_mcmc_samples 10`.
4. Repulsion-guided sampling with `--ligand_diversity_mode conformer`, `pose`, and `environment`.

For protein-nucleic-acid and protein-peptide systems, run:

1. Standard baseline generation.
2. Beta-steered generation, typically using `--beta_z_interface -2`.
3. MCMC trajectory generation with receptor and partner chains explicitly specified.
4. Post-generation contact, overlap, and clash analyses.

---

## Troubleshooting

### 1. The command cannot find `boltz`

Make sure the repository has been installed in editable mode:

```bash
pip install -e .[cuda]
```

If using a virtual environment, activate it before running inference.

### 2. CUDA out-of-memory error

Reduce the number of diffusion samples:

```bash
--diffusion_samples 20
```

or reduce the sampling steps:

```bash
--sampling_steps 100
```

You can also run different guidance modes on separate GPUs using `CUDA_VISIBLE_DEVICES`.

### 3. Incorrect interface mask or empty contacts

Check that:

- `--prior_complex_path` points to the correct receptor-partner complex.
- `--protein_chain_names` and `--partner_chain_names` match the chain IDs in the prior complex.
- The YAML input and prior complex describe the same molecular system.

### 4. MCMC run produces too many outputs

Reduce the number of retained MCMC states:

```bash
--max_mcmc_samples 5
```

or reduce samples per state:

```bash
--diffusion_samples 20
```

---

## License

This repository is released under the MIT License unless otherwise stated. Because ITRG builds on Boltz-2, users should also comply with the license terms of the upstream Boltz-2 project and any third-party dependencies.

---

## Acknowledgements

This repository builds on the Boltz-2 framework:

- **Boltz-2** — Passaro *et al.* 2025, *bioRxiv*. [https://doi.org/10.1101/2025.06.14.659707](https://doi.org/10.1101/2025.06.14.659707)

We thank the Boltz developers for releasing the underlying framework that made this extension possible.

---

## Citation

If you use ITRG in your research, please cite both the ITRG manuscript and the original Boltz-2 paper.

### ITRG

```bibtex
@article{xu2026itrg,
  title   = {ITRG: Protein-Ligand Complex Ensemble Generation with Boltz-2 via Interface Trajectory Conditioning and Repulsive Guidance},
  author  = {Xu, Kai and Tian, Yanan and Zang, Jieying and Liu, Huanxiang and Yao, Xiaojun},
  journal = {Manuscript in preparation},
  year    = {2026}
}
```

### Boltz-2

```bibtex
@article{passaro2025boltz2,
  author  = {Passaro, Saro and Corso, Gabriele and Wohlwend, Jeremy and Reveiz, Mateo and Thaler, Stephan and Somnath, Vignesh Ram and Getz, Noah and Portnoi, Tally and Roy, Julien and Stark, Hannes and Kwabi-Addo, David and Beaini, Dominique and Jaakkola, Tommi and Barzilay, Regina},
  title   = {Boltz-2: Towards Accurate and Efficient Binding Affinity Prediction},
  year    = {2025},
  doi     = {10.1101/2025.06.14.659707},
  journal = {bioRxiv}
}
```
