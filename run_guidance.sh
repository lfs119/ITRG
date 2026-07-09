#!/bin/bash

# baseline
CUDA_VISIBLE_DEVICES=7 boltz predict /home/xukai_cluster/boltz_series/boltz/examples/T4L/4W53.yaml  --use_msa_server  --use_potentials --recycling_steps 10  --sampling_steps 200  --diffusion_samples 100 --override  --seed 42 

# environment  diversity
CUDA_VISIBLE_DEVICES=6 boltz predict /home/xukai_cluster/boltz_series/boltz/examples/T4L/4W53.yaml --enable_protein_patch_mask --use_msa_server  --use_potentials --recycling_steps 10  --sampling_steps 200  --diffusion_samples 100 --override --ligand_diversity_mode  environment --flag env  --seed 42 --prior_complex_path /home/xukai_cluster/boltz_series/boltz/boltz_results_4W53_4W53/predictions/4W53/4W53_model_0.cif

# pose diversity
CUDA_VISIBLE_DEVICES=6 boltz predict /home/xukai_cluster/boltz_series/boltz/examples/T4L/4W53.yaml --enable_protein_patch_mask --use_msa_server  --use_potentials --recycling_steps 10  --sampling_steps 200  --diffusion_samples 100 --override --ligand_diversity_mode  pose --flag pose  --seed 42 --prior_complex_path /home/xukai_cluster/boltz_series/boltz/boltz_results_4W53_4W53/predictions/4W53/4W53_model_0.cif

# conformer diversity
CUDA_VISIBLE_DEVICES=7 boltz predict /home/xukai_cluster/boltz_series/boltz/examples/T4L/4W53.yaml --enable_protein_patch_mask --use_msa_server  --use_potentials --recycling_steps 10  --sampling_steps 200  --diffusion_samples 100 --override --ligand_diversity_mode  conformer --flag conformer  --seed 42 --prior_complex_path /home/xukai_cluster/boltz_series/boltz/boltz_results_4W53_4W53/predictions/4W53/4W53_model_0.cif