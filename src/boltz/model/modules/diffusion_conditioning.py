from __future__ import annotations

import torch
from torch import nn
from torch.nn import Module

from boltz.model.modules.encodersv2 import (
    AtomEncoder,
    PairwiseConditioning,
)

from boltz.interface_mcmc import sample_interface_z_mcmc

class DiffusionConditioning(Module):
    def __init__(
        self,
        token_s: int,
        token_z: int,
        atom_s: int,
        atom_z: int,
        atoms_per_window_queries: int = 32,
        atoms_per_window_keys: int = 128,
        atom_encoder_depth: int = 3,
        atom_encoder_heads: int = 4,
        token_transformer_depth: int = 24,
        token_transformer_heads: int = 8,
        atom_decoder_depth: int = 3,
        atom_decoder_heads: int = 4,
        atom_feature_dim: int = 128,
        conditioning_transition_layers: int = 2,
        use_no_atom_char: bool = False,
        use_atom_backbone_feat: bool = False,
        use_residue_feats_atoms: bool = False,
    ) -> None:
        super().__init__()

        self.pairwise_conditioner = PairwiseConditioning(
            token_z=token_z,
            dim_token_rel_pos_feats=token_z,
            num_transitions=conditioning_transition_layers,
        )

        self.atom_encoder = AtomEncoder(
            atom_s=atom_s,
            atom_z=atom_z,
            token_s=token_s,
            token_z=token_z,
            atoms_per_window_queries=atoms_per_window_queries,
            atoms_per_window_keys=atoms_per_window_keys,
            atom_feature_dim=atom_feature_dim,
            structure_prediction=True,
            use_no_atom_char=use_no_atom_char,
            use_atom_backbone_feat=use_atom_backbone_feat,
            use_residue_feats_atoms=use_residue_feats_atoms,
        )

        self.atom_enc_proj_z = nn.ModuleList()
        for _ in range(atom_encoder_depth):
            self.atom_enc_proj_z.append(
                nn.Sequential(
                    nn.LayerNorm(atom_z),
                    nn.Linear(atom_z, atom_encoder_heads, bias=False),
                )
            )

        self.atom_dec_proj_z = nn.ModuleList()
        for _ in range(atom_decoder_depth):
            self.atom_dec_proj_z.append(
                nn.Sequential(
                    nn.LayerNorm(atom_z),
                    nn.Linear(atom_z, atom_decoder_heads, bias=False),
                )
            )

        self.token_trans_proj_z = nn.ModuleList()
        for _ in range(token_transformer_depth):
            self.token_trans_proj_z.append(
                nn.Sequential(
                    nn.LayerNorm(token_z),
                    nn.Linear(token_z, token_transformer_heads, bias=False),
                )
            )
            
    def steer_forward(
        self,
        z,  # Float['b n n tz']
        relative_position_encoding,  # Float['b n n tz']
        feats,
        steering_args: dict | None = None,
    ):            
            
        if steering_args is not None and steering_args.get("enable_interface_steering", False):
            pad = feats["token_pad_mask"].to(torch.bool)

            mode = steering_args.get("partner_mask_mode", "auto")

            # partner
            if mode == "affinity":
                partner = feats["affinity_token_mask"].to(torch.bool) & pad

            elif mode == "interface":
                partner = feats["interface_token_mask"].to(torch.bool) & pad

            elif mode == "nonprotein":
                partner = (feats["mol_type"] != 0) & pad

            else:
                if "partner_token_mask" in feats:
                    partner = feats["partner_token_mask"].to(torch.bool) & pad
                elif "affinity_token_mask" in feats and feats["affinity_token_mask"].sum() > 0:
                    partner = feats["affinity_token_mask"].to(torch.bool) & pad
                elif "interface_token_mask" in feats and feats["interface_token_mask"].sum() > 0:
                    partner = feats["interface_token_mask"].to(torch.bool) & pad
                else:
                    partner = (feats["mol_type"] != 0) & pad

            
            whole_rec = (feats["mol_type"] == 0) & pad & (~partner)

           
            if "protein_patch_mask" in steering_args and steering_args["protein_patch_mask"] is not None:
                rec = steering_args["protein_patch_mask"].to(torch.bool) & whole_rec
                if not torch.any(rec):
                    rec = whole_rec
            else:
                rec = whole_rec

            #  pair mask
            pl = (partner[:, :, None] & rec[:, None, :]) | (rec[:, :, None] & partner[:, None, :])

            # optional: include partner<->partner block (like affinity does for ligand-ligand)
            include_pp = steering_args.get("include_partner_partner", False)
            if include_pp:
                pp = (partner[:, :, None] & partner[:, None, :])
                if steering_args.get("exclude_partner_diag", True):
                    n = partner.shape[1]
                    diag = torch.eye(n, device=partner.device, dtype=torch.bool)[None, :, :]
                    pp = pp & (~diag)
                pair_mask = pl | pp
            else:
                pair_mask = pl
            
            if steering_args.get("log_interface_mask_stats", False):
                n_pl = int(pl.sum().item()) if pl is not None else 0
                # n_pp = int(pair_mask_pp.sum().item()) if pair_mask_pp is not None else 0              

            
            # final mask: [b, n, n, 1] for broadcasting into z
            M = pair_mask.to(z.dtype).unsqueeze(-1)

            beta = float(steering_args.get("beta_z_interface", 0.0))
            if beta != 0.0:
                # masked scaling
                print(f"******[DiffusionConditioning] interface steering: beta={beta}**********")
                z = z * (1.0 + beta * M)
                
            out = sample_interface_z_mcmc(
                z_base=z,
                pair_mask=pair_mask,
                n_steps=800,
                burn_in=40,
                thin=40,
                step_scale=0.02,
                block_radius=2,
                lambda_anchor=1.0,
                lambda_fast=0.0,     
                fast_score_fn=None,  
                clamp_delta=1.5,
            )

            z_samples = out["samples_z"]       # [S+1,B,N,N,C]
            z_final = out["final_z"]           # [B,N,N,C]
            accept_rate = out["accept_rate"]   # [B]

           
            # cache mask for later bias injection
            feats["_interface_pair_mask"] = pair_mask  # bool [b,n,n]
            
            return z_samples, z_final, accept_rate
        
    
    def forward_single_z(
        self,
        s_trunk,  # Float['b n ts']
        z,        # Float['b n n tz']
        feats,
    ):
        """
        q, c, to_keys, atom_enc_bias, atom_dec_bias, token_trans_bias
        """
        q, c, p, to_keys = self.atom_encoder(
            feats=feats,
            s_trunk=s_trunk,  # Float['b n ts']
            z=z,              # Float['b n n tz']
        )

        atom_enc_bias = []
        for layer in self.atom_enc_proj_z:
            atom_enc_bias.append(layer(p))
        atom_enc_bias = torch.cat(atom_enc_bias, dim=-1)

        atom_dec_bias = []
        for layer in self.atom_dec_proj_z:
            atom_dec_bias.append(layer(p))
        atom_dec_bias = torch.cat(atom_dec_bias, dim=-1)

        token_trans_bias = []
        for layer in self.token_trans_proj_z:
            token_trans_bias.append(layer(z))
        token_trans_bias = torch.cat(token_trans_bias, dim=-1)

        return q, c, to_keys, atom_enc_bias, atom_dec_bias, token_trans_bias


    def forward(
        self,
        s_trunk,  # Float['b n ts']
        z_trunk,  # Float['b n n tz']
        relative_position_encoding,  # Float['b n n tz']
        feats,
        steering_args=None,
    ):
        use_steering = (
            steering_args is not None
            and steering_args.get("enable_interface_steering", False)
        )
        
        z = self.pairwise_conditioner(
                z_trunk,
                relative_position_encoding,
            )  # Float['b n n tz']
            
        if not use_steering:


            return self.forward_single_z(
                s_trunk=s_trunk,
                z=z,
                feats=feats,
            )

        # ---- MCMC / steering  ----
        z_samples, z_final, accept_rate = self.steer_forward(
            z,
            relative_position_encoding,
            feats,
            steering_args,
        )
        #  z_samples: [S+1, B, N, N, Tz]


        num_samples = z_samples.shape[0]
        max_samples = steering_args.get("max_mcmc_samples", 5) if steering_args is not None else num_samples
        if max_samples is not None and max_samples < num_samples:
            z_samples = z_samples[:max_samples]

        sample_outputs = []
        

        for i in range(max_samples):
            z_i = z_samples[i]   # [B, N, N, Tz]

            q_i, c_i, to_keys_i, atom_enc_bias_i, atom_dec_bias_i, token_trans_bias_i = self.forward_single_z(
                s_trunk=s_trunk,
                z=z_i,
                feats=feats,
            )

            sample_outputs.append(
                {
                    "sample_idx": i,
                    "z": z_i,
                    "q": q_i,
                    "c": c_i,
                    "to_keys": to_keys_i,
                    "atom_enc_bias": atom_enc_bias_i,
                    "atom_dec_bias": atom_dec_bias_i,
                    "token_trans_bias": token_trans_bias_i,
                }
            )

        return sample_outputs, z_final, accept_rate

    # def forward(
    #     self,
    #     s_trunk,  # Float['b n ts']
    #     z_trunk,  # Float['b n n tz']
    #     relative_position_encoding,  # Float['b n n tz']
    #     feats,
    #     steering_args: dict | None = None,
    # ):
       
    #     if steering_args is not None and steering_args.get("enable_interface_steering", False):
    #         z_samples, z_final, accept_rate = self.steer_forward(
    #             z_trunk, relative_position_encoding, feats, steering_args)
    #         z = z_samples.squeeze(1)  # (s*b n n tz)
    #         print(f"Interface steering enabled. MCMC accept rate: {accept_rate.mean().item():.4f}")
    #     else:
    #          z = self.pairwise_conditioner(
    #          z_trunk,
    #          relative_position_encoding,) # Float['b n n tz']
        
       
    #     q, c, p, to_keys = self.atom_encoder(
    #         feats=feats,
    #         s_trunk=s_trunk,  # Float['b n ts'],
    #         z=z,  # Float['b n n tz'],
    #     )

    #     atom_enc_bias = []
    #     for layer in self.atom_enc_proj_z:
    #         atom_enc_bias.append(layer(p))
    #     atom_enc_bias = torch.cat(atom_enc_bias, dim=-1)

    #     atom_dec_bias = []
    #     for layer in self.atom_dec_proj_z:
    #         atom_dec_bias.append(layer(p))
    #     atom_dec_bias = torch.cat(atom_dec_bias, dim=-1)

    #     token_trans_bias = []
    #     for layer in self.token_trans_proj_z:
    #         token_trans_bias.append(layer(z))
    #     token_trans_bias = torch.cat(token_trans_bias, dim=-1)
    #     return q, c, to_keys, atom_enc_bias, atom_dec_bias, token_trans_bias
    
 
     
    

