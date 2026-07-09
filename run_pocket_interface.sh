#!/bin/bash

# anchor f1
CUDA_VISIBLE_DEVICES=3 boltz predict /home/xukai_cluster/boltz_series/boltz/examples/affinity_potentials_debug.yaml --enable_protein_patch_mask --use_msa_server  --use_potentials --recycling_steps 10  --sampling_steps 200  --diffusion_samples 100 --override --enable_interface_steering  --beta_z_interface -1 --max_mcmc_samples 1  --seed 42 --flag f1 --protein_chain_names A  --partner_chain_names B --interface_chain B --prior_complex_path /home/xukai_cluster/boltz_series/boltz/boltz_results_affinity/predictions/affinity/affinity_model_0.cif

# anchor f2
CUDA_VISIBLE_DEVICES=3 boltz predict /home/xukai_cluster/boltz_series/boltz/examples/affinity_potentials_debug.yaml --enable_protein_patch_mask --use_msa_server  --use_potentials --recycling_steps 10  --sampling_steps 200  --diffusion_samples 100 --override --enable_interface_steering  --beta_z_interface -2 --max_mcmc_samples 1  --seed 42 --flag f2 --protein_chain_names A  --partner_chain_names B --interface_chain B --prior_complex_path /home/xukai_cluster/boltz_series/boltz/boltz_results_affinity/predictions/affinity/affinity_model_0.cif
          
# MCMC f2
CUDA_VISIBLE_DEVICES=3 boltz predict /home/xukai_cluster/boltz_series/boltz/examples/affinity_potentials_debug.yaml --enable_protein_patch_mask --use_msa_server  --use_potentials --recycling_steps 10  --sampling_steps 200  --diffusion_samples 100 --override --enable_interface_steering  --beta_z_interface -2 --max_mcmc_samples 10  --seed 42 --flag MCMCf2 --protein_chain_names A  --partner_chain_names B --interface_chain B --prior_complex_path /home/xukai_cluster/boltz_series/boltz/boltz_results_affinity/predictions/affinity/affinity_model_0.cif

# MCMC f1
CUDA_VISIBLE_DEVICES=3 boltz predict /home/xukai_cluster/boltz_series/boltz/examples/affinity_potentials_debug.yaml --enable_protein_patch_mask --use_msa_server  --use_potentials --recycling_steps 10  --sampling_steps 200  --diffusion_samples 100 --override --enable_interface_steering  --beta_z_interface -1 --max_mcmc_samples 10  --seed 42 --flag MCMCf1 --protein_chain_names A  --partner_chain_names B --interface_chain B --prior_complex_path /home/xukai_cluster/boltz_series/boltz/boltz_results_affinity/predictions/affinity/affinity_model_0.cif