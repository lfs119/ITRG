from abc import ABC, abstractmethod
import math
from typing import Optional, Dict, Any, Set, List, Union

import torch
import torch.nn.functional as F
import numpy as np
from boltz.data import const
from boltz.model.potentials.schedules import (
    ParameterSchedule,
    ExponentialInterpolation,
    PiecewiseStepFunction,
)
from boltz.model.loss.diffusionv2 import weighted_rigid_align


class Potential(ABC):
    def __init__(
        self,
        parameters: Optional[
            Dict[str, Union[ParameterSchedule, float, int, bool]]
        ] = None,
    ):
        self.parameters = parameters

    def compute(self, coords, feats, parameters):
        index, args, com_args, ref_args, operator_args = self.compute_args(
            feats, parameters
        )

        if index.shape[1] == 0:
            return torch.zeros(coords.shape[:-2], device=coords.device)

        if com_args is not None:
            com_index, atom_pad_mask = com_args
            unpad_com_index = com_index[atom_pad_mask]
            unpad_coords = coords[..., atom_pad_mask, :]
            coords = torch.zeros(
                (*unpad_coords.shape[:-2], unpad_com_index.max() + 1, 3),
                device=coords.device,
            ).scatter_reduce(
                -2,
                unpad_com_index.unsqueeze(-1).expand_as(unpad_coords),
                unpad_coords,
                "mean",
            )
        else:
            com_index, atom_pad_mask = None, None

        if ref_args is not None:
            ref_coords, ref_mask, ref_atom_index, ref_token_index = ref_args
            coords = coords[..., ref_atom_index, :]
        else:
            ref_coords, ref_mask, ref_atom_index, ref_token_index = (
                None,
                None,
                None,
                None,
            )

        if operator_args is not None:
            negation_mask, union_index = operator_args
        else:
            negation_mask, union_index = None, None

        value = self.compute_variable(
            coords,
            index,
            ref_coords=ref_coords,
            ref_mask=ref_mask,
            compute_gradient=False,
        )
        energy = self.compute_function(
            value, *args, negation_mask=negation_mask, compute_derivative=False
        )

        if union_index is not None:
            neg_exp_energy = torch.exp(-1 * parameters["union_lambda"] * energy)
            Z = torch.zeros(
                (*energy.shape[:-1], union_index.max() + 1), device=union_index.device
            ).scatter_reduce(
                -1,
                union_index.expand_as(neg_exp_energy),
                neg_exp_energy,
                "sum",
            )
            softmax_energy = neg_exp_energy / Z[..., union_index]
            softmax_energy[Z[..., union_index] == 0] = 0
            return (energy * softmax_energy).sum(dim=-1)

        return energy.sum(dim=tuple(range(1, energy.dim())))

    def compute_gradient(self, coords, feats, parameters):
        index, args, com_args, ref_args, operator_args = self.compute_args(
            feats, parameters
        )
        if index.shape[1] == 0:
            return torch.zeros_like(coords)

        if com_args is not None:
            com_index, atom_pad_mask = com_args
            unpad_coords = coords[..., atom_pad_mask, :]
            unpad_com_index = com_index[atom_pad_mask]
            coords = torch.zeros(
                (*unpad_coords.shape[:-2], unpad_com_index.max() + 1, 3),
                device=coords.device,
            ).scatter_reduce(
                -2,
                unpad_com_index.unsqueeze(-1).expand_as(unpad_coords),
                unpad_coords,
                "mean",
            )
            com_counts = torch.bincount(com_index[atom_pad_mask])
        else:
            com_index, atom_pad_mask = None, None

        if ref_args is not None:
            ref_coords, ref_mask, ref_atom_index, ref_token_index = ref_args
            coords = coords[..., ref_atom_index, :]
        else:
            ref_coords, ref_mask, ref_atom_index, ref_token_index = (
                None,
                None,
                None,
                None,
            )

        if operator_args is not None:
            negation_mask, union_index = operator_args
        else:
            negation_mask, union_index = None, None

        value, grad_value = self.compute_variable(
            coords,
            index,
            ref_coords=ref_coords,
            ref_mask=ref_mask,
            compute_gradient=True,
        )
        energy, dEnergy = self.compute_function(
            value, 
            *args, negation_mask=negation_mask, compute_derivative=True
        )
        if union_index is not None:
            neg_exp_energy = torch.exp(-1 * parameters["union_lambda"] * energy)
            Z = torch.zeros(
                (*energy.shape[:-1], union_index.max() + 1), device=union_index.device
            ).scatter_reduce(
                -1,
                union_index.expand_as(energy),
                neg_exp_energy,
                "sum",
            )
            softmax_energy = neg_exp_energy / Z[..., union_index]
            softmax_energy[Z[..., union_index] == 0] = 0
            f = torch.zeros(
                (*energy.shape[:-1], union_index.max() + 1), device=union_index.device
            ).scatter_reduce(
                -1,
                union_index.expand_as(energy),
                energy * softmax_energy,
                "sum",
            )
            dSoftmax = (
                dEnergy
                * softmax_energy
                * (1 + parameters["union_lambda"] * (energy - f[..., union_index]))
            )
            prod = dSoftmax.tile(grad_value.shape[-3]).unsqueeze(
                -1
            ) * grad_value.flatten(start_dim=-3, end_dim=-2)
            if prod.dim() > 3:
                prod = prod.sum(dim=list(range(1, prod.dim() - 2)))
            grad_atom = torch.zeros_like(coords).scatter_reduce(
                -2,
                index.flatten(start_dim=0, end_dim=1)
                .unsqueeze(-1)
                .expand((*coords.shape[:-2], -1, 3)),
                prod,
                "sum",
            )
        else:
            prod = dEnergy.tile(grad_value.shape[-3]).unsqueeze(
                -1
            ) * grad_value.flatten(start_dim=-3, end_dim=-2)
            if prod.dim() > 3:
                prod = prod.sum(dim=list(range(1, prod.dim() - 2)))
            grad_atom = torch.zeros_like(coords).scatter_reduce(
                -2,
                index.flatten(start_dim=0, end_dim=1)
                .unsqueeze(-1)
                .expand((*coords.shape[:-2], -1, 3)),  # 9 x 516 x 3
                prod,
                "sum",
            )

        if com_index is not None:
            grad_atom = grad_atom[..., com_index, :]
        elif ref_token_index is not None:
            grad_atom = grad_atom[..., ref_token_index, :]

        return grad_atom

    def compute_parameters(self, t):
        if self.parameters is None:
            return None
        parameters = {
            name: parameter
            if not isinstance(parameter, ParameterSchedule)
            else parameter.compute(t)
            for name, parameter in self.parameters.items()
        }
        return parameters

    @abstractmethod
    def compute_function(
        self, value, *args, negation_mask=None, compute_derivative=False
    ):
        raise NotImplementedError

    @abstractmethod
    def compute_variable(self, coords, index, compute_gradient=False):
        raise NotImplementedError

    @abstractmethod
    def compute_args(self, t, feats, **parameters):
        raise NotImplementedError

    def get_reference_coords(self, feats, parameters):
        return None, None


class FlatBottomPotential(Potential):
    def compute_function(
        self,
        value,
        k,
        lower_bounds,
        upper_bounds,
        negation_mask=None,
        compute_derivative=False,
    ):
        if lower_bounds is None:
            lower_bounds = torch.full_like(value, float("-inf"))
        if upper_bounds is None:
            upper_bounds = torch.full_like(value, float("inf"))
        lower_bounds = lower_bounds.expand_as(value).clone()
        upper_bounds = upper_bounds.expand_as(value).clone()

        if negation_mask is not None:
            unbounded_below_mask = torch.isneginf(lower_bounds)
            unbounded_above_mask = torch.isposinf(upper_bounds)
            unbounded_mask = unbounded_below_mask + unbounded_above_mask
            assert torch.all(unbounded_mask + negation_mask)
            lower_bounds[~unbounded_above_mask * ~negation_mask] = upper_bounds[
                ~unbounded_above_mask * ~negation_mask
            ]
            upper_bounds[~unbounded_above_mask * ~negation_mask] = float("inf")
            upper_bounds[~unbounded_below_mask * ~negation_mask] = lower_bounds[
                ~unbounded_below_mask * ~negation_mask
            ]
            lower_bounds[~unbounded_below_mask * ~negation_mask] = float("-inf")

        neg_overflow_mask = value < lower_bounds
        pos_overflow_mask = value > upper_bounds

        energy = torch.zeros_like(value)
        energy[neg_overflow_mask] = (k * (lower_bounds - value))[neg_overflow_mask]
        energy[pos_overflow_mask] = (k * (value - upper_bounds))[pos_overflow_mask]
        if not compute_derivative:
            return energy

        dEnergy = torch.zeros_like(value)
        dEnergy[neg_overflow_mask] = (
            -1 * k.expand_as(neg_overflow_mask)[neg_overflow_mask]
        )
        dEnergy[pos_overflow_mask] = (
            1 * k.expand_as(pos_overflow_mask)[pos_overflow_mask]
        )

        return energy, dEnergy


class ReferencePotential(Potential):
    def compute_variable(
        self, coords, index, ref_coords, ref_mask, compute_gradient=False
    ):
        aligned_ref_coords = weighted_rigid_align(
            ref_coords.float(),
            coords[:, index].float(),
            ref_mask,
            ref_mask,
        )

        r = coords[:, index] - aligned_ref_coords
        r_norm = torch.linalg.norm(r, dim=-1)

        if not compute_gradient:
            return r_norm

        r_hat = r / r_norm.unsqueeze(-1)
        grad = (r_hat * ref_mask.unsqueeze(-1)).unsqueeze(1)
        return r_norm, grad


class DistancePotential(Potential):
    def compute_variable(
        self, coords, index, ref_coords=None, ref_mask=None, compute_gradient=False
    ):
        r_ij = coords.index_select(-2, index[0]) - coords.index_select(-2, index[1])
        r_ij_norm = torch.linalg.norm(r_ij, dim=-1)
        r_hat_ij = r_ij / r_ij_norm.unsqueeze(-1)

        if not compute_gradient:
            return r_ij_norm

        grad_i = r_hat_ij
        grad_j = -1 * r_hat_ij
        grad = torch.stack((grad_i, grad_j), dim=1)
        return r_ij_norm, grad


class DihedralPotential(Potential):
    def compute_variable(
        self, coords, index, ref_coords=None, ref_mask=None, compute_gradient=False
    ):
        r_ij = coords.index_select(-2, index[0]) - coords.index_select(-2, index[1])
        r_kj = coords.index_select(-2, index[2]) - coords.index_select(-2, index[1])
        r_kl = coords.index_select(-2, index[2]) - coords.index_select(-2, index[3])

        n_ijk = torch.cross(r_ij, r_kj, dim=-1)
        n_jkl = torch.cross(r_kj, r_kl, dim=-1)

        r_kj_norm = torch.linalg.norm(r_kj, dim=-1)
        n_ijk_norm = torch.linalg.norm(n_ijk, dim=-1)
        n_jkl_norm = torch.linalg.norm(n_jkl, dim=-1)

        sign_phi = torch.sign(
            r_kj.unsqueeze(-2) @ torch.cross(n_ijk, n_jkl, dim=-1).unsqueeze(-1)
        ).squeeze(-1, -2)
        phi = sign_phi * torch.arccos(
            torch.clamp(
                (n_ijk.unsqueeze(-2) @ n_jkl.unsqueeze(-1)).squeeze(-1, -2)
                / (n_ijk_norm * n_jkl_norm),
                -1 + 1e-8,
                1 - 1e-8,
            )
        )

        if not compute_gradient:
            return phi

        a = (
            (r_ij.unsqueeze(-2) @ r_kj.unsqueeze(-1)).squeeze(-1, -2) / (r_kj_norm**2)
        ).unsqueeze(-1)
        b = (
            (r_kl.unsqueeze(-2) @ r_kj.unsqueeze(-1)).squeeze(-1, -2) / (r_kj_norm**2)
        ).unsqueeze(-1)

        grad_i = n_ijk * (r_kj_norm / n_ijk_norm**2).unsqueeze(-1)
        grad_l = -1 * n_jkl * (r_kj_norm / n_jkl_norm**2).unsqueeze(-1)
        grad_j = (a - 1) * grad_i - b * grad_l
        grad_k = (b - 1) * grad_l - a * grad_i
        grad = torch.stack((grad_i, grad_j, grad_k, grad_l), dim=1)
        return phi, grad


class AbsDihedralPotential(DihedralPotential):
    def compute_variable(
        self, coords, index, ref_coords=None, ref_mask=None, compute_gradient=False
    ):
        if not compute_gradient:
            phi = super().compute_variable(
                coords, index, compute_gradient=compute_gradient
            )
            phi = torch.abs(phi)
            return phi

        phi, grad = super().compute_variable(
            coords, index, compute_gradient=compute_gradient
        )
        grad[(phi < 0)[..., None, :, None].expand_as(grad)] *= -1
        phi = torch.abs(phi)

        return phi, grad


class PoseBustersPotential(FlatBottomPotential, DistancePotential):
    def compute_args(self, feats, parameters):
        pair_index = feats["rdkit_bounds_index"][0]
        lower_bounds = feats["rdkit_lower_bounds"][0].clone()
        upper_bounds = feats["rdkit_upper_bounds"][0].clone()
        bond_mask = feats["rdkit_bounds_bond_mask"][0]
        angle_mask = feats["rdkit_bounds_angle_mask"][0]

        lower_bounds[bond_mask * ~angle_mask] *= 1.0 - parameters["bond_buffer"]
        upper_bounds[bond_mask * ~angle_mask] *= 1.0 + parameters["bond_buffer"]
        lower_bounds[~bond_mask * angle_mask] *= 1.0 - parameters["angle_buffer"]
        upper_bounds[~bond_mask * angle_mask] *= 1.0 + parameters["angle_buffer"]
        lower_bounds[bond_mask * angle_mask] *= 1.0 - min(
            parameters["bond_buffer"], parameters["angle_buffer"]
        )
        upper_bounds[bond_mask * angle_mask] *= 1.0 + min(
            parameters["bond_buffer"], parameters["angle_buffer"]
        )
        lower_bounds[~bond_mask * ~angle_mask] *= 1.0 - parameters["clash_buffer"]
        upper_bounds[~bond_mask * ~angle_mask] = float("inf")

        vdw_radii = torch.zeros(
            const.num_elements, dtype=torch.float32, device=pair_index.device
        )
        vdw_radii[1:119] = torch.tensor(
            const.vdw_radii, dtype=torch.float32, device=pair_index.device
        )
        atom_vdw_radii = (
            feats["ref_element"].float() @ vdw_radii.unsqueeze(-1)
        ).squeeze(-1)[0]
        bond_cutoffs = 0.35 + atom_vdw_radii[pair_index].mean(dim=0)
        lower_bounds[~bond_mask] = torch.max(lower_bounds[~bond_mask], bond_cutoffs[~bond_mask])
        upper_bounds[bond_mask] = torch.min(upper_bounds[bond_mask], bond_cutoffs[bond_mask])

        k = torch.ones_like(lower_bounds)

        return pair_index, (k, lower_bounds, upper_bounds), None, None, None


class ConnectionsPotential(FlatBottomPotential, DistancePotential):
    def compute_args(self, feats, parameters):
        pair_index = feats["connected_atom_index"][0]
        lower_bounds = None
        upper_bounds = torch.full(
            (pair_index.shape[1],), parameters["buffer"], device=pair_index.device
        )
        k = torch.ones_like(upper_bounds)

        return pair_index, (k, lower_bounds, upper_bounds), None, None, None


class VDWOverlapPotential(FlatBottomPotential, DistancePotential):
    def compute_args(self, feats, parameters):
        atom_chain_id = (
            torch.bmm(
                feats["atom_to_token"].float(), feats["asym_id"].unsqueeze(-1).float()
            )
            .squeeze(-1)
            .long()
        )[0]
        atom_pad_mask = feats["atom_pad_mask"][0].bool()
        chain_sizes = torch.bincount(atom_chain_id[atom_pad_mask])
        single_ion_mask = (chain_sizes > 1)[atom_chain_id]

        vdw_radii = torch.zeros(
            const.num_elements, dtype=torch.float32, device=atom_chain_id.device
        )
        vdw_radii[1:119] = torch.tensor(
            const.vdw_radii, dtype=torch.float32, device=atom_chain_id.device
        )
        atom_vdw_radii = (
            feats["ref_element"].float() @ vdw_radii.unsqueeze(-1)
        ).squeeze(-1)[0]

        pair_index = torch.triu_indices(
            atom_chain_id.shape[0],
            atom_chain_id.shape[0],
            1,
            device=atom_chain_id.device,
        )

        pair_pad_mask = atom_pad_mask[pair_index].all(dim=0)
        pair_ion_mask = single_ion_mask[pair_index[0]] * single_ion_mask[pair_index[1]]

        num_chains = atom_chain_id.max() + 1
        connected_chain_index = feats["connected_chain_index"][0]
        connected_chain_matrix = torch.eye(
            num_chains, device=atom_chain_id.device, dtype=torch.bool
        )
        connected_chain_matrix[connected_chain_index[0], connected_chain_index[1]] = (
            True
        )
        connected_chain_matrix[connected_chain_index[1], connected_chain_index[0]] = (
            True
        )
        connected_chain_mask = connected_chain_matrix[
            atom_chain_id[pair_index[0]], atom_chain_id[pair_index[1]]
        ]

        pair_index = pair_index[
            :, pair_pad_mask * pair_ion_mask * ~connected_chain_mask
        ]

        lower_bounds = atom_vdw_radii[pair_index].sum(dim=0) * (
            1.0 - parameters["buffer"]
        )
        upper_bounds = None
        k = torch.ones_like(lower_bounds)

        return pair_index, (k, lower_bounds, upper_bounds), None, None, None


class SymmetricChainCOMPotential(FlatBottomPotential, DistancePotential):
    def compute_args(self, feats, parameters):
        atom_chain_id = (
            torch.bmm(
                feats["atom_to_token"].float(), feats["asym_id"].unsqueeze(-1).float()
            )
            .squeeze(-1)
            .long()
        )[0]
        atom_pad_mask = feats["atom_pad_mask"][0].bool()
        chain_sizes = torch.bincount(atom_chain_id[atom_pad_mask])
        single_ion_mask = chain_sizes > 1

        pair_index = feats["symmetric_chain_index"][0]
        pair_ion_mask = single_ion_mask[pair_index[0]] * single_ion_mask[pair_index[1]]
        pair_index = pair_index[:, pair_ion_mask]
        lower_bounds = torch.full(
            (pair_index.shape[1],),
            parameters["buffer"],
            dtype=torch.float32,
            device=pair_index.device,
        )
        upper_bounds = None
        k = torch.ones_like(lower_bounds)

        return (
            pair_index,
            (k, lower_bounds, upper_bounds),
            (atom_chain_id, atom_pad_mask),
            None,
            None,
        )


class StereoBondPotential(FlatBottomPotential, AbsDihedralPotential):
    def compute_args(self, feats, parameters):
        stereo_bond_index = feats["stereo_bond_index"][0]
        stereo_bond_orientations = feats["stereo_bond_orientations"][0].bool()

        lower_bounds = torch.zeros(
            stereo_bond_orientations.shape, device=stereo_bond_orientations.device
        )
        upper_bounds = torch.zeros(
            stereo_bond_orientations.shape, device=stereo_bond_orientations.device
        )
        lower_bounds[stereo_bond_orientations] = torch.pi - parameters["buffer"]
        upper_bounds[stereo_bond_orientations] = float("inf")
        lower_bounds[~stereo_bond_orientations] = float("-inf")
        upper_bounds[~stereo_bond_orientations] = parameters["buffer"]

        k = torch.ones_like(lower_bounds)

        return stereo_bond_index, (k, lower_bounds, upper_bounds), None, None, None


class ChiralAtomPotential(FlatBottomPotential, DihedralPotential):
    def compute_args(self, feats, parameters):
        chiral_atom_index = feats["chiral_atom_index"][0]
        chiral_atom_orientations = feats["chiral_atom_orientations"][0].bool()

        lower_bounds = torch.zeros(
            chiral_atom_orientations.shape, device=chiral_atom_orientations.device
        )
        upper_bounds = torch.zeros(
            chiral_atom_orientations.shape, device=chiral_atom_orientations.device
        )
        lower_bounds[chiral_atom_orientations] = parameters["buffer"]
        upper_bounds[chiral_atom_orientations] = float("inf")
        upper_bounds[~chiral_atom_orientations] = -1 * parameters["buffer"]
        lower_bounds[~chiral_atom_orientations] = float("-inf")

        k = torch.ones_like(lower_bounds)
        return chiral_atom_index, (k, lower_bounds, upper_bounds), None, None, None


class PlanarBondPotential(FlatBottomPotential, AbsDihedralPotential):
    def compute_args(self, feats, parameters):
        double_bond_index = feats["planar_bond_index"][0].T
        double_bond_improper_index = torch.tensor(
            [
                [1, 2, 3, 0],
                [4, 5, 0, 3],
            ],
            device=double_bond_index.device,
        ).T
        improper_index = (
            double_bond_index[:, double_bond_improper_index]
            .swapaxes(0, 1)
            .flatten(start_dim=1)
        )
        lower_bounds = None
        upper_bounds = torch.full(
            (improper_index.shape[1],),
            parameters["buffer"],
            device=improper_index.device,
        )
        k = torch.ones_like(upper_bounds)

        return improper_index, (k, lower_bounds, upper_bounds), None, None, None


class TemplateReferencePotential(FlatBottomPotential, ReferencePotential):
    def compute_args(self, feats, parameters):
        if "template_mask_cb" not in feats or "template_force" not in feats:
            return torch.empty([1, 0]), None, None, None, None

        template_mask = feats["template_mask_cb"][feats["template_force"]]
        if template_mask.shape[0] == 0:
            return torch.empty([1, 0]), None, None, None, None

        ref_coords = feats["template_cb"][feats["template_force"]].clone()
        ref_mask = feats["template_mask_cb"][feats["template_force"]].clone()
        ref_atom_index = (
            torch.bmm(
                feats["token_to_rep_atom"].float(),
                torch.arange(
                    feats["atom_pad_mask"].shape[1],
                    device=feats["atom_pad_mask"].device,
                    dtype=torch.float32,
                )[None, :, None],
            )
            .squeeze(-1)
            .long()
        )[0]
        ref_token_index = (
            torch.bmm(
                feats["atom_to_token"].float(),
                feats["token_index"].unsqueeze(-1).float(),
            )
            .squeeze(-1)
            .long()
        )[0]

        index = torch.arange(
            template_mask.shape[-1], dtype=torch.long, device=template_mask.device
        )[None]
        upper_bounds = torch.full(
            template_mask.shape, float("inf"), device=index.device, dtype=torch.float32
        )
        ref_idxs = torch.argwhere(template_mask).T
        upper_bounds[ref_idxs.unbind()] = feats["template_force_threshold"][
            feats["template_force"]
        ][ref_idxs[0]]

        lower_bounds = None
        k = torch.ones_like(upper_bounds)
        return (
            index,
            (k, lower_bounds, upper_bounds),
            None,
            (ref_coords, ref_mask, ref_atom_index, ref_token_index),
            None,
        )


class ContactPotentital(FlatBottomPotential, DistancePotential):
    def compute_args(self, feats, parameters):
        index = feats["contact_pair_index"][0]
        union_index = feats["contact_union_index"][0]
        negation_mask = feats["contact_negation_mask"][0]
        lower_bounds = None
        upper_bounds = feats["contact_thresholds"][0].clone()
        k = torch.ones_like(upper_bounds)
        return (
            index,
            (k, lower_bounds, upper_bounds),
            None,
            None,
            (negation_mask, union_index),
        )


class LigandDiversityPotentialBase(Potential):
    """
    Base class for batch-level diversity potentials.

    Assumption:
        batch == 1 in the data-loader sense
        B == num_samples in coords.shape == [B, N, 3]

    This class overrides compute/compute_gradient directly because the standard
    Potential path in this file is built around per-index local terms, while
    diversity here couples different samples in the leading dimension.
    """

    def _pair_distance_matrix(self, coords, feats, parameters):
        raise NotImplementedError

    def _pair_energy(self, pair_dist, parameters):
        """
        pair_dist: [B, B]
        energy for pair (i, j) is active only when pair_dist < min_diversity
        """
        margin = float(parameters["min_diversity"])
        x = F.relu(margin - pair_dist)

        power = int(parameters.get("power", 2))
        if power == 1:
            pair_energy = x
        elif power == 2:
            pair_energy = x * x
        else:
            pair_energy = x.pow(power)

        eye = torch.eye(pair_dist.shape[0], device=pair_dist.device, dtype=torch.bool)
        pair_energy = pair_energy.masked_fill(eye, 0.0)
        return pair_energy

    def compute(self, coords, feats, parameters):
        if coords.dim() != 3:
            raise ValueError(
                f"{self.__class__.__name__} expects coords with shape [B, N, 3], got {tuple(coords.shape)}"
            )

        B = coords.shape[0]
        if B <= 1:
            return torch.zeros(B, device=coords.device, dtype=coords.dtype)

        pair_dist = self._pair_distance_matrix(coords, feats, parameters)  # [B, B]
        pair_energy = self._pair_energy(pair_dist, parameters)             # [B, B]

        # per-sample energy
        energy = pair_energy.sum(dim=-1) / max(B - 1, 1)
        return energy

    def compute_gradient(self, coords, feats, parameters):
        
        # print("grad enabled:", torch.is_grad_enabled())
        # print("inference mode:", torch.is_inference_mode_enabled())
        
        if coords.shape[0] <= 1:
            return torch.zeros_like(coords)

        with torch.inference_mode(False):
            with torch.enable_grad():
                coords_req = coords.detach().clone().requires_grad_(True)
                energy = self.compute(coords_req, feats, parameters)
                
                # print("coords_req.requires_grad:", coords_req.requires_grad)
                # print("energy.requires_grad:", energy.requires_grad)
                # print("energy.grad_fn:", energy.grad_fn)

                if not energy.requires_grad:
                    return torch.zeros_like(coords)

                grad = torch.autograd.grad(
                    energy.sum(),
                    coords_req,
                    retain_graph=False,
                    create_graph=False,
                    allow_unused=False,
                )[0]

        return torch.zeros_like(coords) if grad is None else grad

    # Dummy methods to satisfy the abstract interface.
    # They are not used because compute/compute_gradient are overridden.
    def compute_function(
        self, value, *args, negation_mask=None, compute_derivative=False
    ):
        raise NotImplementedError(
            f"{self.__class__.__name__} overrides compute() directly."
        )

    def compute_variable(self, coords, index, ref_coords=None, ref_mask=None, compute_gradient=False):
        raise NotImplementedError(
            f"{self.__class__.__name__} overrides compute() directly."
        )

    def compute_args(self, feats, parameters):
        device = feats["atom_pad_mask"].device
        return torch.empty((1, 0), dtype=torch.long, device=device), None, None, None, None

    # ---------- helpers ----------
    def _get_index(self, feats, index_key, mask_key=None):
        if index_key not in feats:
            raise KeyError(f"Missing feats['{index_key}'] for {self.__class__.__name__}")

        idx = feats[index_key]

        if mask_key is not None and mask_key in feats:
            local_mask = feats[mask_key].bool()
            idx = idx[local_mask]

        return idx

    def _safe_masked_rmsd(self, x, y, mask=None, eps=1e-8):
        """
        x, y: [L, 3]
        mask: [L] bool or None
        """
        diff2 = ((x - y) ** 2).sum(dim=-1)  # [L]
        if mask is None:
            return torch.sqrt(diff2.mean() + eps)

        w = mask.float()
        denom = w.sum().clamp_min(1.0)
        return torch.sqrt((diff2 * w).sum() / denom + eps)


class LigandConformerDiversityPotential(LigandDiversityPotentialBase):
    """
    Encourage conformer diversity across B generated samples by comparing
    rigid-motion-invariant ligand internal descriptors.

    Descriptor:
        upper-triangle of ligand internal pair distances
        or precomputed feats["ligand_internal_pair_index"] if provided.

    Good for:
        conformer diversity
    Not sensitive to:
        global translation / global rotation of the ligand
    """

    def _ligand_descriptor(self, coords, feats, parameters):
      
        lig_coords = coords[:, feats['ligand_atom_index'], :]  # [B, L, 3]
        B, L, _ = lig_coords.shape

        if L < 2:
            return lig_coords.new_zeros((B, 1))

        if "ligand_internal_pair_index" in feats:
            # local ligand indices in [0, L)
            pair_index = feats["ligand_internal_pair_index"][0].long()
            d = torch.linalg.norm(
                lig_coords[:, pair_index[0], :] - lig_coords[:, pair_index[1], :],
                dim=-1,
            )  # [B, P]
        else:
            dmat = torch.cdist(lig_coords, lig_coords, p=2)  # [B, L, L]
            iu = torch.triu_indices(L, L, offset=1, device=coords.device)
            d = dmat[:, iu[0], iu[1]]  # [B, P]

        # normalize so the descriptor distance scale does not explode with ligand size
        d = d / math.sqrt(max(d.shape[-1], 1))
        return d

    def _pair_distance_matrix(self, coords, feats, parameters):
        desc = self._ligand_descriptor(coords, feats, parameters)  # [B, D]
        pair_dist = torch.cdist(desc, desc, p=2)                   # [B, B]
        
        # print("pair_dist.requires_grad:", pair_dist.requires_grad)
        # print("pair_dist.grad_fn:", pair_dist.grad_fn)

        return pair_dist


class LigandAlignedPoseDiversityPotential(LigandDiversityPotentialBase):
    """
    Encourage diversity across B generated samples by:
        1) aligning protein atoms between samples
        2) computing ligand RMSD in the aligned protein frame
    """

    def _kabsch_transform_batched(self, mobile, target, mask=None, eps=1e-8):
        """
        Batched row-vector rigid transform:
            aligned = mobile @ R + t

        mobile, target: [N, P, 3]
        mask:
            - None
            - [P]
            - [N, P]
        Returns:
            R: [N, 3, 3]
            t: [N, 3]
        """
        N, P, _ = mobile.shape
        dtype = mobile.dtype
        device = mobile.device

        if mask is None:
            w = torch.ones((N, P), device=device, dtype=dtype)
        else:
            if mask.ndim == 1:
                w = mask.to(device=device, dtype=dtype)[None, :].expand(N, -1)
            elif mask.ndim == 2:
                w = mask.to(device=device, dtype=dtype)
            else:
                raise ValueError(f"mask must be None, [P], or [N, P], got shape={mask.shape}")

        w_sum = w.sum(dim=1, keepdim=True).clamp_min(1.0)
        w = w / w_sum  # [N, P]

        w3 = w[..., None]  # [N, P, 1]

        mobile_center = (mobile * w3).sum(dim=1)  # [N, 3]
        target_center = (target * w3).sum(dim=1)  # [N, 3]

        Xm = mobile - mobile_center[:, None, :]   # [N, P, 3]
        Ym = target - target_center[:, None, :]   # [N, P, 3]

        # H = (Xm * w)^T @ Ym
        H = torch.matmul((Xm * w3).transpose(1, 2), Ym)  # [N, 3, 3]

        # mixed precision 
        svd_dtype = torch.float32 if H.dtype in (torch.float16, torch.bfloat16) else H.dtype
        H_svd = H.to(svd_dtype)

        U, S, Vh = torch.linalg.svd(H_svd, full_matrices=False)

        # reflection-safe rotation
        V = Vh.transpose(-2, -1)
        Ut = U.transpose(-2, -1)

        det_val = torch.det(V @ Ut)
        M = torch.eye(3, device=H.device, dtype=svd_dtype).unsqueeze(0).repeat(N, 1, 1)
        M[:, -1, -1] = torch.where(
            det_val < 0,
            torch.full_like(det_val, -1.0),
            torch.full_like(det_val,  1.0),
        )

        R = V @ M @ Ut  # [N, 3, 3]
        R = R.to(dtype)

        # row-vector convention: aligned = mobile @ R + t
        t = target_center - torch.bmm(mobile_center[:, None, :], R).squeeze(1)  # [N, 3]

        return R, t

    def _safe_masked_rmsd_batched(self, x, y, mask=None, eps=1e-8):
        """
        x, y: [N, L, 3]
        mask:
            - None
            - [L]
            - [N, L]
        return: [N]
        """
        d2 = ((x - y) ** 2).sum(dim=-1)  # [N, L]

        if mask is None:
            msd = d2.mean(dim=1)
        else:
            if mask.ndim == 1:
                w = mask.to(device=x.device, dtype=x.dtype)[None, :].expand(x.shape[0], -1)
            elif mask.ndim == 2:
                w = mask.to(device=x.device, dtype=x.dtype)
            else:
                raise ValueError(f"mask must be None, [L], or [N, L], got shape={mask.shape}")

            denom = w.sum(dim=1).clamp_min(1.0)
            msd = (d2 * w).sum(dim=1) / denom

        return torch.sqrt(msd + eps)

    def _pair_distance_matrix(self, coords, feats, parameters):
        if "interface_partner_atom_indices" in feats:
            lig_idx = feats["interface_partner_atom_indices"]
        else:
            lig_idx = feats["ligand_atom_index"]
        prot_idx = feats["interface_protein_atom_indices"]

        lig_coords = coords[:, lig_idx, :]   # [B, L, 3]
        prot_coords = coords[:, prot_idx, :] # [B, P, 3]

        B = coords.shape[0]
        L = lig_coords.shape[1]
        P = prot_coords.shape[1]

        if L == 0:
            return coords.new_zeros((B, B))
        if B <= 1:
            return coords.new_zeros((B, B))
        if P < 3:
            raise ValueError(
                "LigandAlignedPoseDiversityPotential needs at least 3 protein alignment atoms."
            )

        prot_mask = None
        if "protein_atom_mask" in feats:
            prot_mask = feats["protein_atom_mask"][0].bool()[:P]

        lig_mask = None
        if "ligand_atom_mask" in feats:
            lig_mask = feats["ligand_atom_mask"][0].bool()[:L]

        eps = float(parameters.get("eps", 1e-8))
        pair_batch_size = int(parameters.get("pair_batch_size", 4096))

        pair_dist = coords.new_zeros((B, B))

        
        idx_i, idx_j = torch.triu_indices(B, B, offset=1, device=coords.device)
        num_pairs = idx_i.numel()

        if num_pairs == 0:
            return pair_dist

        rmsd_vals = coords.new_empty(num_pairs)

        for start in range(0, num_pairs, pair_batch_size):
            end = min(start + pair_batch_size, num_pairs)
            ii = idx_i[start:end]
            jj = idx_j[start:end]

            # align sample j onto sample i using protein atoms
            R, t = self._kabsch_transform_batched(
                mobile=prot_coords[jj],   # [n, P, 3]
                target=prot_coords[ii],   # [n, P, 3]
                mask=prot_mask,
                eps=eps,
            )

            lig_j_aligned = torch.bmm(lig_coords[jj], R) + t[:, None, :]  # [n, L, 3]

            rmsd_ij = self._safe_masked_rmsd_batched(
                lig_coords[ii],
                lig_j_aligned,
                mask=lig_mask,
                eps=eps,
            )

            rmsd_vals[start:end] = rmsd_ij

        pair_dist[idx_i, idx_j] = rmsd_vals
        pair_dist[idx_j, idx_i] = rmsd_vals
        return pair_dist

class LigandEnvironmentDiversityPotential(LigandDiversityPotentialBase):
    """
    Dynamic-shell environment diversity potential.

    This implementation uses:
      - interface_protein_atom_indices as a fixed candidate protein pool
      - ligand_atom_index as the current ligand center
      - dynamic 3.0 / 4.5 / 6.0 A shells built from current coordinates

    Core idea:
      1) select a fixed candidate protein pool (prefer interface protein atoms)
      2) use the current ligand atoms as the center
      3) compute protein-to-ligand soft-nearest distances
      4) build dynamic shells:
            shell-1:   0.0 ~ 3.0 A
            shell-2:   3.0 ~ 4.5 A
            shell-3:   4.5 ~ 6.0 A
      5) summarize each shell into a descriptor
      6) compare descriptors across samples in the batch

    Preferred feats:
      - interface_protein_atom_indices
      - interface_protein_atom_type_onehot
      - ligand_atom_index

    Optional feats:
      - partner_atom_type_onehot
      - ligand_atom_type_onehot
      - ligand_atom_mask
      - interface_protein_atom_mask

    Fallback feats:
      - protein_atom_index
    """

    def _maybe_squeeze_first(self, x):
        if isinstance(x, torch.Tensor):
            if x.dim() in (2, 3) and x.shape[0] == 1:
                return x[0]
            return x
        if isinstance(x, np.ndarray):
            if x.ndim in (2, 3) and x.shape[0] == 1:
                return x[0]
            return x
        return x

    def _to_long_index(self, x, device):
        x = self._maybe_squeeze_first(x)
        if isinstance(x, torch.Tensor):
            return x.to(device=device, dtype=torch.long)
        return torch.as_tensor(x, device=device, dtype=torch.long)

    def _to_float_tensor(self, x, device, dtype):
        x = self._maybe_squeeze_first(x)
        if isinstance(x, torch.Tensor):
            return x.to(device=device, dtype=dtype)
        return torch.as_tensor(x, device=device, dtype=dtype)

    def _softmin_over_atoms(self, d, tau):
        """
        Soft minimum over the last dimension.

        d: [..., M]
        return: [...]
        """
        tau = max(float(tau), 1e-6)
        return -tau * torch.logsumexp(-d / tau, dim=-1)

    def _typed_softmin_to_ligand(self, d_pl, lig_type_onehot, tau):
        """
        Type-resolved soft-nearest distance from protein atoms to ligand atoms.

        Args:
          d_pl: [B, P, L]
          lig_type_onehot: [L, C_l]

        Returns:
          d_typed: [B, P, C_l]
        """
        tau = max(float(tau), 1e-6)

        # [B, P, L]
        expv = torch.exp(-d_pl / tau)

        # [B, P, C_l]
        num = torch.einsum("bpl,lc->bpc", expv, lig_type_onehot)
        d_typed = -tau * torch.log(num.clamp_min(1e-12))

        lig_type_presence = lig_type_onehot.sum(dim=0) > 0  # [C_l]
        if (~lig_type_presence).any():
            d_typed[:, :, ~lig_type_presence] = 1e6

        return d_typed

    def _build_dynamic_shells(self, d_env, radii, tau):
        """
        Build cumulative and ring-shell gates from protein-to-ligand distances.

        Args:
          d_env: [B, P]
          radii: tensor([3.0, 4.5, 6.0])
          tau: smoothing width

        Returns:
          cumulative: [B, 3, P]
          rings:      [B, 3, P]
              ring-0: 0.0 ~ 3.0 A
              ring-1: 3.0 ~ 4.5 A
              ring-2: 4.5 ~ 6.0 A
        """
        tau = max(float(tau), 1e-6)

        g1 = torch.sigmoid((radii[0] - d_env) / tau)  # within 3.0
        g2 = torch.sigmoid((radii[1] - d_env) / tau)  # within 4.5
        g3 = torch.sigmoid((radii[2] - d_env) / tau)  # within 6.0

        cumulative = torch.stack([g1, g2, g3], dim=1)

        shell1 = g1
        shell2 = torch.clamp(g2 - g1, min=0.0)
        shell3 = torch.clamp(g3 - g2, min=0.0)

        rings = torch.stack([shell1, shell2, shell3], dim=1)
        return cumulative, rings

    def _safe_topk_mean(self, x, k, dim=-1):
        k = int(min(max(k, 1), x.shape[dim]))
        vals, _ = torch.topk(x, k=k, dim=dim)
        return vals.mean(dim=dim)

    def _environment_descriptor(self, coords, feats, parameters):
        device = coords.device
        dtype = coords.dtype

        # ------------------------------------------------------------------
        # 1) Select candidate protein pool and current ligand atoms
        # ------------------------------------------------------------------
        if "interface_protein_atom_indices" in feats:
            prot_idx = self._to_long_index(feats["interface_protein_atom_indices"], device)
        else:
            prot_idx = self._to_long_index(feats["protein_atom_index"], device)
        
        if "interface_partner_atom_indices" in feats:
            lig_idx = self._to_long_index(feats["interface_partner_atom_indices"], device)
        else:
            lig_idx = self._to_long_index(feats["ligand_atom_index"], device)

        prot_coords = coords[:, prot_idx, :]  # [B, P, 3]
        lig_coords = coords[:, lig_idx, :]    # [B, L, 3]

        # Optional masks
        if "interface_protein_atom_mask" in feats:
            prot_mask = self._to_float_tensor(
                feats["interface_protein_atom_mask"], device=device, dtype=dtype
            ).bool()
            prot_coords = prot_coords[:, prot_mask, :]
        else:
            prot_mask = None

        if "ligand_atom_mask" in feats:
            lig_mask = self._to_float_tensor(
                feats["ligand_atom_mask"], device=device, dtype=dtype
            ).bool()
            lig_coords = lig_coords[:, lig_mask, :]
        else:
            lig_mask = None

        B, P, _ = prot_coords.shape
        _, L, _ = lig_coords.shape

        if P == 0 or L == 0:
            return coords.new_zeros((B, 1))

        # ------------------------------------------------------------------
        # 2) Chemical channels
        # ------------------------------------------------------------------
        # Protein type channels should align with the candidate pool.
        if "interface_protein_atom_type_onehot" in feats and "interface_protein_atom_indices" in feats:
            prot_type_onehot = self._to_float_tensor(
                feats["interface_protein_atom_type_onehot"], device=device, dtype=dtype
            )
            if prot_mask is not None:
                prot_type_onehot = prot_type_onehot[prot_mask]
        else:
            prot_type_onehot = torch.ones((P, 1), device=device, dtype=dtype)

        # Ligand type channels: prefer full ligand-aligned onehot if available.
        if "partner_atom_type_onehot" in feats:
            lig_type_onehot = self._to_float_tensor(
                feats["partner_atom_type_onehot"], device=device, dtype=dtype
            )
            if lig_mask is not None:
                lig_type_onehot = lig_type_onehot[lig_mask]
        elif "ligand_atom_type_onehot" in feats:
            lig_type_onehot = self._to_float_tensor(
                feats["ligand_atom_type_onehot"], device=device, dtype=dtype
            )
            if lig_mask is not None:
                lig_type_onehot = lig_type_onehot[lig_mask]
        else:
            lig_type_onehot = torch.ones((L, 1), device=device, dtype=dtype)

        if prot_type_onehot.shape[0] != P:
            raise ValueError(
                f"interface_protein_atom_type_onehot rows ({prot_type_onehot.shape[0]}) != candidate protein atoms ({P})"
            )
        if lig_type_onehot.shape[0] != L:
            raise ValueError(
                f"ligand/partner atom type rows ({lig_type_onehot.shape[0]}) != ligand atoms ({L})"
            )

        # ------------------------------------------------------------------
        # 3) Parameters
        # ------------------------------------------------------------------
        radii = parameters.get("radii", [3.0, 4.5, 6.0])
        if len(radii) != 3:
            raise ValueError("Expected radii=[3.0, 4.5, 6.0] or another length-3 list.")

        pair_rbf_centers = parameters.get("pair_rbf_centers", [2.5, 3.0, 3.5, 4.0, 5.0, 6.0])

        softmin_tau = parameters.get("softmin_tau", 0.10)
        typed_softmin_tau = parameters.get("typed_softmin_tau", 0.10)
        radius_tau = parameters.get("radius_tau", 0.10)
        pair_rbf_sigma = parameters.get("pair_rbf_sigma", 0.20)

        contact_sharpness = parameters.get("contact_sharpness", 2.0)
        rbf_sharpness = parameters.get("rbf_sharpness", 1.5)

        include_shell_density = parameters.get("include_shell_density", True)
        include_shell_moments = parameters.get("include_shell_moments", True)
        include_protein_type_shell = parameters.get("include_protein_type_shell", True)
        include_pair_type_contact = parameters.get("include_pair_type_contact", True)
        include_pair_type_rbf = parameters.get("include_pair_type_rbf", False)
        include_ligtype_nearest = parameters.get("include_ligtype_nearest", True)

        topk_contacts = parameters.get("topk_contacts", 8)
        topk_pair_rbf = parameters.get("topk_pair_rbf", 12)

        radii = torch.as_tensor(radii, device=device, dtype=dtype)  # [3]
        pair_rbf_centers = torch.as_tensor(pair_rbf_centers, device=device, dtype=dtype)  # [K]

        radius_tau = max(float(radius_tau), 1e-6)
        pair_rbf_sigma = max(float(pair_rbf_sigma), 1e-6)

        # ------------------------------------------------------------------
        # 4) Geometry
        # ------------------------------------------------------------------
        # Full distance matrix between candidate protein atoms and current ligand atoms.
        # [B, P, L]
        d_pl = torch.cdist(prot_coords, lig_coords, p=2)

        # Protein-to-ligand soft nearest distance: [B, P]
        d_env = self._softmin_over_atoms(d_pl, tau=softmin_tau)

        # Dynamic shells over candidate protein atoms.
        _, ring_gate = self._build_dynamic_shells(d_env, radii, tau=radius_tau)  # [B, 3, P]

        desc_parts = []

        # ------------------------------------------------------------------
        # 5) Shell density and shell moments
        # ------------------------------------------------------------------
        shell_norm = ring_gate.sum(dim=-1, keepdim=True).clamp_min(1e-8)  # [B, 3, 1]

        if include_shell_density:
            shell_density = ring_gate.mean(dim=-1)  # [B, 3]
            desc_parts.append(shell_density)

        if include_shell_moments:
            shell_mean = (ring_gate * d_env[:, None, :]).sum(dim=-1) / shell_norm.squeeze(-1)  # [B, 3]
            centered2 = (d_env[:, None, :] - shell_mean[:, :, None]) ** 2
            shell_std = torch.sqrt(
                (ring_gate * centered2).sum(dim=-1) / shell_norm.squeeze(-1) + 1e-8
            )  # [B, 3]
            desc_parts.extend([shell_mean, shell_std])

        # ------------------------------------------------------------------
        # 6) Protein type composition in each dynamic shell
        # ------------------------------------------------------------------
        if include_protein_type_shell:
            # [B, 3, C_p]
            shell_type = torch.einsum("bsp,pc->bsc", ring_gate, prot_type_onehot) / shell_norm
            desc_parts.append(shell_type.flatten(start_dim=1))

        # ------------------------------------------------------------------
        # 7) Pair-type contact descriptor using current dynamic geometry
        # ------------------------------------------------------------------
        if include_pair_type_contact:
            # [B, 3, P, L]
            pair_contact = torch.sigmoid(
                (radii[None, :, None, None] - d_pl[:, None, :, :]) / radius_tau
            ).pow(contact_sharpness)

            # [B, 3, C_p, C_l]
            desc_pair_contact = torch.einsum(
                "bspl,pc,ld->bscd",
                pair_contact,
                prot_type_onehot,
                lig_type_onehot,
            ) / math.sqrt(max(P * L, 1))

            desc_parts.append(desc_pair_contact.flatten(start_dim=1))

            # [B, 3]
            flat_contact = pair_contact.reshape(B, 3, P * L)
            desc_parts.append(self._safe_topk_mean(flat_contact, k=topk_contacts, dim=-1))

        # ------------------------------------------------------------------
        # 8) Pair-type RBF descriptor over current dynamic geometry
        # ------------------------------------------------------------------
        if include_pair_type_rbf:
            # [B, K, P, L]
            pair_rbf = torch.exp(
                -0.5 * ((d_pl[:, None, :, :] - pair_rbf_centers[None, :, None, None]) / pair_rbf_sigma) ** 2
            ).pow(rbf_sharpness)

            # [B, K, C_p, C_l]
            desc_pair_rbf = torch.einsum(
                "bkpl,pc,ld->bkcd",
                pair_rbf,
                prot_type_onehot,
                lig_type_onehot,
            ) / math.sqrt(max(P * L, 1))

            desc_parts.append(desc_pair_rbf.flatten(start_dim=1))

            flat_rbf = pair_rbf.reshape(B, len(pair_rbf_centers), P * L)
            desc_parts.append(self._safe_topk_mean(flat_rbf, k=topk_pair_rbf, dim=-1))

        # ------------------------------------------------------------------
        # 9) Ligand-type-resolved nearest environment in each shell
        # ------------------------------------------------------------------
        if include_ligtype_nearest:
            # [B, P, C_l]
            d_env_ligtype = self._typed_softmin_to_ligand(
                d_pl=d_pl,
                lig_type_onehot=lig_type_onehot,
                tau=typed_softmin_tau,
            )

            # [B, 3, P, C_l]
            ring_gate_ligtype = ring_gate[:, :, :, None]

            # [B, 3, C_l]
            shell_ligtype_mean = (
                (ring_gate_ligtype * d_env_ligtype[:, None, :, :]).sum(dim=2)
                / shell_norm
            )
            desc_parts.append(shell_ligtype_mean.flatten(start_dim=1))

        # ------------------------------------------------------------------
        # 10) Final descriptor
        # ------------------------------------------------------------------
        desc = torch.cat(desc_parts, dim=-1)  # [B, D]
        desc = desc / math.sqrt(max(desc.shape[-1], 1))
        return desc

    def _pair_distance_matrix(self, coords, feats, parameters):
        desc = self._environment_descriptor(coords, feats, parameters)  # [B, D]
        pair_dist = torch.cdist(desc, desc, p=1)  # L1 is usually sharper than L2 here
        return pair_dist
    


class LigandEnvironmentPairDiversityPotential(LigandDiversityPotentialBase):
    """
    Batch-level diversity potential based on a fixed interface contact-pair map.

    Core idea:
      - Keep the identity of interface protein atoms and interface partner atoms.
      - If interface_contact_pair_index is available, build the descriptor directly
        from the true prior-contact pairs instead of the full Cartesian product.
      - Use atom-type information only as an auxiliary weighting signal.

    Preferred feats:
      - interface_protein_atom_indices
      - interface_partner_atom_indices
      - interface_contact_pair_index

    Optional feats:
      - interface_protein_atom_type_onehot
      - interface_partner_atom_type_onehot
      - atom_type_vocab

    Fallback feats:
      - protein_atom_index
      - ligand_atom_index
    """

    def _maybe_squeeze_first(self, x):
        if isinstance(x, torch.Tensor):
            if x.dim() in (2, 3) and x.shape[0] == 1:
                return x[0]
            return x
        if isinstance(x, np.ndarray):
            if x.ndim in (2, 3) and x.shape[0] == 1:
                return x[0]
            return x
        return x

    def _to_long_index(self, x, device):
        x = self._maybe_squeeze_first(x)
        if isinstance(x, torch.Tensor):
            return x.to(device=device, dtype=torch.long)
        return torch.as_tensor(x, device=device, dtype=torch.long)

    def _to_float_tensor(self, x, device, dtype):
        x = self._maybe_squeeze_first(x)
        if isinstance(x, torch.Tensor):
            return x.to(device=device, dtype=dtype)
        return torch.as_tensor(x, device=device, dtype=dtype)

    def _to_vocab_list(self, x):
        x = self._maybe_squeeze_first(x)
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        if isinstance(x, np.ndarray):
            return [str(v) for v in x.tolist()]
        if isinstance(x, (list, tuple)):
            return [str(v) for v in x]
        return None

    def _safe_topk_mean(self, x, k, dim=-1):
        k = int(min(max(k, 1), x.shape[dim]))
        vals, _ = torch.topk(x, k=k, dim=dim)
        return vals.mean(dim=dim)

    def _build_pair_weight_vector(
        self,
        feats,
        pair_index,
        M,
        device,
        dtype,
        pair_type_gain: float = 0.5,
        metal_gain: float = 0.5,
        halogen_gain: float = 0.25,
    ):
        """
        Build an auxiliary pair-weight vector W[m] from atom types.

        pair_index: [2, M], local indices into
          - interface_protein_atom_type_onehot
          - interface_partner_atom_type_onehot

        Returns:
          pair_weight: [M]
        """
        if (
            "interface_protein_atom_type_onehot" not in feats
            or "interface_partner_atom_type_onehot" not in feats
            or "atom_type_vocab" not in feats
        ):
            return torch.ones((M,), device=device, dtype=dtype)

        prot_oh_all = self._to_float_tensor(
            feats["interface_protein_atom_type_onehot"], device=device, dtype=dtype
        )  # [P, C]
        lig_oh_all = self._to_float_tensor(
            feats["interface_partner_atom_type_onehot"], device=device, dtype=dtype
        )  # [L, C]
        vocab = self._to_vocab_list(feats["atom_type_vocab"])

        if vocab is None:
            return torch.ones((M,), device=device, dtype=dtype)

        if pair_index.shape[1] != M:
            return torch.ones((M,), device=device, dtype=dtype)

        prot_oh = prot_oh_all[pair_index[0]]  # [M, C]
        lig_oh = lig_oh_all[pair_index[1]]    # [M, C]

        vocab_to_idx = {str(v): i for i, v in enumerate(vocab)}

        def idxs(names):
            return [vocab_to_idx[n] for n in names if n in vocab_to_idx]

        carbon_idx = idxs(["C"])
        unk_idx = idxs(["UNK"])
        metal_idx = idxs([
            "Li", "Na", "K", "Rb", "Cs", "Mg", "Ca", "Sr", "Ba",
            "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Cd", "Hg",
            "V", "Cr", "Mo", "W"
        ])
        halogen_idx = idxs(["F", "Cl", "Br", "I"])

        all_idx = list(range(len(vocab)))
        hetero_idx = [i for i in all_idx if i not in set(carbon_idx + unk_idx)]

        def channel_sum(onehot, index_list):
            if len(index_list) == 0:
                return torch.zeros(onehot.shape[0], device=device, dtype=dtype)
            return onehot[:, index_list].sum(dim=-1)

        prot_hetero = channel_sum(prot_oh, hetero_idx)  # [M]
        lig_hetero = channel_sum(lig_oh, hetero_idx)    # [M]
        prot_metal = channel_sum(prot_oh, metal_idx)    # [M]
        lig_metal = channel_sum(lig_oh, metal_idx)      # [M]
        prot_hal = channel_sum(prot_oh, halogen_idx)    # [M]
        lig_hal = channel_sum(lig_oh, halogen_idx)      # [M]

        pair_weight = torch.ones((M,), device=device, dtype=dtype)
        pair_weight = pair_weight + pair_type_gain * (prot_hetero * lig_hetero)
        pair_weight = pair_weight + metal_gain * (prot_metal + lig_metal)
        pair_weight = pair_weight + halogen_gain * (prot_hal + lig_hal)

        return pair_weight

    def _environment_descriptor(self, coords, feats, parameters):
        device = coords.device
        dtype = coords.dtype

        # ------------------------------------------------------------------
        # 1) Select interface atoms
        # ------------------------------------------------------------------
        if "interface_protein_atom_indices" in feats:
            prot_idx = self._to_long_index(feats["interface_protein_atom_indices"], device)
        else:
            prot_idx = self._to_long_index(feats["protein_atom_index"], device)

        if "interface_partner_atom_indices" in feats:
            lig_idx = self._to_long_index(feats["interface_partner_atom_indices"], device)
        else:
            lig_idx = self._to_long_index(feats["ligand_atom_index"], device)

        prot_coords_all = coords[:, prot_idx, :]  # [B, P, 3]
        lig_coords_all = coords[:, lig_idx, :]    # [B, L, 3]

        B, P, _ = prot_coords_all.shape
        _, L, _ = lig_coords_all.shape

        if P == 0 or L == 0:
            return coords.new_zeros((B, 1))

        # ------------------------------------------------------------------
        # 2) Parameters
        # ------------------------------------------------------------------
        radii = parameters.get("radii", [2.5, 3.0, 3.5, 4.5])
        rbf_centers = parameters.get("rbf_centers", [2.2, 2.6, 3.0, 3.4, 3.8, 4.4])

        radius_tau = parameters.get("radius_tau", 0.08)
        rbf_sigma = parameters.get("rbf_sigma", 0.18)
        nearest_tau = parameters.get("nearest_tau", 0.08)

        contact_sharpness = parameters.get("contact_sharpness", 2.0)
        rbf_sharpness = parameters.get("rbf_sharpness", 1.5)

        topk_contacts = parameters.get("topk_contacts", 8)
        topk_rbf = parameters.get("topk_rbf", 12)

        include_pair_contact = parameters.get("include_pair_contact", True)
        include_pair_rbf = parameters.get("include_pair_rbf", True)
        include_nearest_stats = parameters.get("include_nearest_stats", True)

        use_pair_type_weight = parameters.get("use_pair_type_weight", True)
        pair_type_gain = parameters.get("pair_type_gain", 0.5)
        metal_gain = parameters.get("metal_gain", 0.5)
        halogen_gain = parameters.get("halogen_gain", 0.25)

        radii = torch.as_tensor(radii, device=device, dtype=dtype)         # [R]
        centers = torch.as_tensor(rbf_centers, device=device, dtype=dtype) # [K]

        radius_tau = max(float(radius_tau), 1e-6)
        rbf_sigma = max(float(rbf_sigma), 1e-6)
        nearest_tau = max(float(nearest_tau), 1e-6)

        desc_parts = []

        # ------------------------------------------------------------------
        # 3) True prior-contact-pair mode
        # ------------------------------------------------------------------
        if "interface_contact_pair_index" in feats:
            pair_index = self._to_long_index(feats["interface_contact_pair_index"], device)  # [2, M]

            if pair_index.numel() == 0:
                return coords.new_zeros((B, 1))

            prot_coords_pair = prot_coords_all[:, pair_index[0], :]  # [B, M, 3]
            lig_coords_pair = lig_coords_all[:, pair_index[1], :]    # [B, M, 3]

            # True pair distances only: [B, M]
            d_pair = torch.linalg.norm(prot_coords_pair - lig_coords_pair, dim=-1)

            M = d_pair.shape[1]
            if M == 0:
                return coords.new_zeros((B, 1))

            if use_pair_type_weight:
                pair_weight = self._build_pair_weight_vector(
                    feats=feats,
                    pair_index=pair_index,
                    M=M,
                    device=device,
                    dtype=dtype,
                    pair_type_gain=pair_type_gain,
                    metal_gain=metal_gain,
                    halogen_gain=halogen_gain,
                )  # [M]
            else:
                pair_weight = torch.ones((M,), device=device, dtype=dtype)

            # 3a) Multi-radius soft contact over true contact pairs
            if include_pair_contact:
                # [B, R, M]
                pair_contact = torch.sigmoid(
                    (radii[None, :, None] - d_pair[:, None, :]) / radius_tau
                ).pow(contact_sharpness)

                pair_contact = pair_contact * pair_weight[None, None, :]

                # Keep pair identity
                desc_parts.append(pair_contact.flatten(start_dim=1))

                # Top-k summary per shell: [B, R]
                desc_parts.append(
                    self._safe_topk_mean(pair_contact, k=topk_contacts, dim=-1)
                )

            # 3b) Pair-distance RBF over true contact pairs
            if include_pair_rbf:
                # [B, K, M]
                pair_rbf = torch.exp(
                    -0.5 * ((d_pair[:, None, :] - centers[None, :, None]) / rbf_sigma) ** 2
                ).pow(rbf_sharpness)

                pair_rbf = pair_rbf * pair_weight[None, None, :]

                desc_parts.append(pair_rbf.flatten(start_dim=1))
                desc_parts.append(
                    self._safe_topk_mean(pair_rbf, k=topk_rbf, dim=-1)
                )

            # 3c) Pair-distance summaries
            if include_nearest_stats:
                desc_parts.append(d_pair)                              # [B, M]
                desc_parts.append(torch.sort(d_pair, dim=-1).values)  # [B, M]

                inv_close = 1.0 / (d_pair + 1e-6)  # [B, M]
                desc_parts.append(
                    self._safe_topk_mean(inv_close, k=max(topk_contacts, 4), dim=-1).unsqueeze(-1)
                )

            desc = torch.cat(desc_parts, dim=-1)  # [B, D]
            desc = desc / math.sqrt(max(desc.shape[-1], 1))
            return desc

        # ------------------------------------------------------------------
        # 4) Fallback: full interface Cartesian-product mode
        # ------------------------------------------------------------------
        d_pl = torch.cdist(prot_coords_all, lig_coords_all, p=2)  # [B, P, L]

        if use_pair_type_weight:
            pair_weight = self._build_pair_weight_matrix(
                feats=feats,
                P=P,
                L=L,
                device=device,
                dtype=dtype,
                pair_type_gain=pair_type_gain,
                metal_gain=metal_gain,
                halogen_gain=halogen_gain,
            )  # [P, L]
        else:
            pair_weight = torch.ones((P, L), device=device, dtype=dtype)

        if include_pair_contact:
            pair_contact = torch.sigmoid(
                (radii[None, :, None, None] - d_pl[:, None, :, :]) / radius_tau
            ).pow(contact_sharpness)
            pair_contact = pair_contact * pair_weight[None, None, :, :]
            desc_parts.append(pair_contact.flatten(start_dim=1))

            flat_contact = pair_contact.reshape(B, len(radii), P * L)
            desc_parts.append(
                self._safe_topk_mean(flat_contact, k=topk_contacts, dim=-1)
            )

        if include_pair_rbf:
            pair_rbf = torch.exp(
                -0.5 * ((d_pl[:, None, :, :] - centers[None, :, None, None]) / rbf_sigma) ** 2
            ).pow(rbf_sharpness)
            pair_rbf = pair_rbf * pair_weight[None, None, :, :]
            desc_parts.append(pair_rbf.flatten(start_dim=1))

            flat_rbf = pair_rbf.reshape(B, len(centers), P * L)
            desc_parts.append(
                self._safe_topk_mean(flat_rbf, k=topk_rbf, dim=-1)
            )

        if include_nearest_stats:
            prot_nearest = self._protein_centered_nearest_to_ligand(d_pl, tau=nearest_tau)  # [B, P]
            lig_nearest = self._ligand_centered_nearest_to_protein(d_pl, tau=nearest_tau)   # [B, L]

            desc_parts.append(prot_nearest)
            desc_parts.append(torch.sort(prot_nearest, dim=-1).values)
            desc_parts.append(lig_nearest)
            desc_parts.append(torch.sort(lig_nearest, dim=-1).values)

            inv_close = 1.0 / (d_pl + 1e-6)
            flat_inv_close = inv_close.reshape(B, P * L)
            desc_parts.append(
                self._safe_topk_mean(flat_inv_close, k=max(topk_contacts, 4), dim=-1).unsqueeze(-1)
            )

        desc = torch.cat(desc_parts, dim=-1)  # [B, D]
        desc = desc / math.sqrt(max(desc.shape[-1], 1))
        return desc

    def _build_pair_weight_matrix(
        self,
        feats,
        P,
        L,
        device,
        dtype,
        pair_type_gain: float = 0.5,
        metal_gain: float = 0.5,
        halogen_gain: float = 0.25,
    ):
        """
        Fallback full [P, L] pair-weight matrix for Cartesian-product mode.
        """
        if (
            "interface_protein_atom_type_onehot" not in feats
            or "interface_partner_atom_type_onehot" not in feats
            or "atom_type_vocab" not in feats
        ):
            return torch.ones((P, L), device=device, dtype=dtype)

        prot_oh = self._to_float_tensor(
            feats["interface_protein_atom_type_onehot"], device=device, dtype=dtype
        )  # [P, C]
        lig_oh = self._to_float_tensor(
            feats["interface_partner_atom_type_onehot"], device=device, dtype=dtype
        )  # [L, C]
        vocab = self._to_vocab_list(feats["atom_type_vocab"])

        if prot_oh.shape[0] != P or lig_oh.shape[0] != L or vocab is None:
            return torch.ones((P, L), device=device, dtype=dtype)

        vocab_to_idx = {str(v): i for i, v in enumerate(vocab)}

        def idxs(names):
            return [vocab_to_idx[n] for n in names if n in vocab_to_idx]

        carbon_idx = idxs(["C"])
        unk_idx = idxs(["UNK"])
        metal_idx = idxs([
            "Li", "Na", "K", "Rb", "Cs", "Mg", "Ca", "Sr", "Ba",
            "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Cd", "Hg",
            "V", "Cr", "Mo", "W"
        ])
        halogen_idx = idxs(["F", "Cl", "Br", "I"])

        all_idx = list(range(len(vocab)))
        hetero_idx = [i for i in all_idx if i not in set(carbon_idx + unk_idx)]

        def channel_sum(onehot, index_list):
            if len(index_list) == 0:
                return torch.zeros(onehot.shape[0], device=device, dtype=dtype)
            return onehot[:, index_list].sum(dim=-1)

        prot_hetero = channel_sum(prot_oh, hetero_idx)   # [P]
        lig_hetero = channel_sum(lig_oh, hetero_idx)     # [L]
        prot_metal = channel_sum(prot_oh, metal_idx)     # [P]
        lig_metal = channel_sum(lig_oh, metal_idx)       # [L]
        prot_hal = channel_sum(prot_oh, halogen_idx)     # [P]
        lig_hal = channel_sum(lig_oh, halogen_idx)       # [L]

        pair_weight = torch.ones((P, L), device=device, dtype=dtype)
        pair_weight = pair_weight + pair_type_gain * (prot_hetero[:, None] * lig_hetero[None, :])
        pair_weight = pair_weight + metal_gain * (prot_metal[:, None] + lig_metal[None, :])
        pair_weight = pair_weight + halogen_gain * (prot_hal[:, None] + lig_hal[None, :])

        return pair_weight

    def _protein_centered_nearest_to_ligand(self, d_pl, tau):
        tau = max(float(tau), 1e-6)
        return -tau * torch.logsumexp(-d_pl / tau, dim=-1)

    def _ligand_centered_nearest_to_protein(self, d_pl, tau):
        tau = max(float(tau), 1e-6)
        return -tau * torch.logsumexp(-d_pl / tau, dim=-2)

    def _pair_distance_matrix(self, coords, feats, parameters):
        desc = self._environment_descriptor(coords, feats, parameters)  # [B, D]
        # pair_dist = torch.cdist(desc, desc, p=1)  # L1 for sharper separation
        return desc


        return pair_dist
    
def get_potentials(steering_args, boltz2=False):
    potentials = []
    if steering_args["fk_steering"] or steering_args["physical_guidance_update"]:
        potentials.extend(
            [
                SymmetricChainCOMPotential(
                    parameters={
                        "guidance_interval": 4,
                        "guidance_weight": 0.5
                        if steering_args["physical_guidance_update"]
                        else 0.0,
                        "resampling_weight": 0.5,
                        "buffer": ExponentialInterpolation(
                            start=1.0, end=5.0, alpha=-2.0
                        ),
                    }
                ),
                VDWOverlapPotential(
                    parameters={
                        "guidance_interval": 5,
                        "guidance_weight": (
                            PiecewiseStepFunction(thresholds=[0.4], values=[0.125, 0.0])
                            if steering_args["physical_guidance_update"]
                            else 0.0
                        ),
                        "resampling_weight": PiecewiseStepFunction(
                            thresholds=[0.6], values=[0.01, 0.0]
                        ),
                        "buffer": 0.225,
                    }
                ),
                ConnectionsPotential(
                    parameters={
                        "guidance_interval": 1,
                        "guidance_weight": 0.15
                        if steering_args["physical_guidance_update"]
                        else 0.0,
                        "resampling_weight": 1.0,
                        "buffer": 2.0,
                    }
                ),
                PoseBustersPotential(
                    parameters={
                        "guidance_interval": 1,
                        "guidance_weight": 0.01
                        if steering_args["physical_guidance_update"]
                        else 0.0,
                        "resampling_weight": 0.1,
                        "bond_buffer": 0.125,
                        "angle_buffer": 0.125,
                        "clash_buffer": 0.10,
                    }
                ),
                ChiralAtomPotential(
                    parameters={
                        "guidance_interval": 1,
                        "guidance_weight": 0.1
                        if steering_args["physical_guidance_update"]
                        else 0.0,
                        "resampling_weight": 1.0,
                        "buffer": 0.52360,
                    }
                ),
                StereoBondPotential(
                    parameters={
                        "guidance_interval": 1,
                        "guidance_weight": 0.05
                        if steering_args["physical_guidance_update"]
                        else 0.0,
                        "resampling_weight": 1.0,
                        "buffer": 0.52360,
                    }
                ),
                PlanarBondPotential(
                    parameters={
                        "guidance_interval": 1,
                        "guidance_weight": 0.05
                        if steering_args["physical_guidance_update"]
                        else 0.0,
                        "resampling_weight": 1.0,
                        "buffer": 0.26180,
                    }
                ),
            ]
        )
    if boltz2 and (
        steering_args["fk_steering"] or steering_args["contact_guidance_update"]
    ):
        potentials.extend(
            [
                ContactPotentital(
                    parameters={
                        "guidance_interval": 4,
                        "guidance_weight": (
                            PiecewiseStepFunction(
                                thresholds=[0.25, 0.75], values=[0.0, 0.5, 1.0]
                            )
                            if steering_args["contact_guidance_update"]
                            else 0.0
                        ),
                        "resampling_weight": 1.0,
                        "union_lambda": ExponentialInterpolation(
                            start=8.0, end=0.0, alpha=-2.0
                        ),
                    }
                ),
                TemplateReferencePotential(
                    parameters={
                        "guidance_interval": 2,
                        "guidance_weight": 0.1
                        if steering_args["contact_guidance_update"]
                        else 0.0,
                        "resampling_weight": 1.0,
                    }
                ),
            ]
        )
    # if steering_args.get("ligand_diversity_guidance", False):
    #     potentials.append(
    #     LigandConformerDiversityPotential(
    #         parameters={
    #             "guidance_interval": 2,
    #             "guidance_weight": PiecewiseStepFunction(
    #                 thresholds=[0.35, 0.75],
    #                 values=[0.05, 0.02, 0.0],
    #             ),
    #             "resampling_weight": 0.0,
    #             "min_diversity": 0.5,
    #             "square": True,
    #         }
    #     )
    # )
    
    diversity_mode = steering_args.get("ligand_diversity_mode", None)

    if diversity_mode == "conformer":
        potentials.append(
            LigandConformerDiversityPotential(
                parameters={
                    "guidance_interval": 5,
                    # "guidance_weight": steering_args.get("ligand_diversity_guidance_weight", 0.02),
                    "resampling_weight": 0.0,
                    "guidance_weight": PiecewiseStepFunction(
                    thresholds=[0.35, 0.75],
                    values=[0.8, 0.4, 0.0],
                ),
                    "min_diversity": steering_args.get("ligand_diversity_min_diversity", 0.50),
                    "power": 2,
                }
            )
        )
    elif diversity_mode == "pose":
        potentials.append(
            LigandAlignedPoseDiversityPotential(
                parameters={
                    "guidance_interval": 2,
                    # "guidance_weight": steering_args.get("ligand_diversity_guidance_weight", 0.02),
                    "guidance_weight": PiecewiseStepFunction(
                    thresholds=[0.35, 0.75],
                    values=[0.8, 0.4, 0.2],
                ),
                    "resampling_weight": 0.0,
                    "min_diversity": steering_args.get("ligand_diversity_min_diversity", 3.0),
                    "power": 2,
                    "eps": 1e-8,
                }
            )
        )
    elif diversity_mode == "environment":
       potentials.append(
    LigandEnvironmentDiversityPotential(
        parameters={
            "guidance_interval": 2,
            "guidance_weight": PiecewiseStepFunction(
                thresholds=[0.35, 0.75],
                values=[0.8, 0.5, 0.2],
            ),
            "resampling_weight": 0.0,
            "min_diversity": steering_args.get("ligand_diversity_min_diversity", 0.8),
            "power": 2,

            # Dynamic shells around ligand
            "radii": steering_args.get("ligand_env_radii", [3.0, 4.5, 6.0]),

            # Pair RBF over dynamic geometry
            "pair_rbf_centers": steering_args.get(
                "ligand_env_pair_rbf_centers",
                [2.5, 3.0, 3.5, 4.0, 5.0, 6.0],
            ),

            # Smoother than before to avoid saturation
            "softmin_tau": steering_args.get("ligand_env_softmin_tau", 0.10),
            "typed_softmin_tau": steering_args.get("ligand_env_typed_softmin_tau", 0.10),
            "radius_tau": steering_args.get("ligand_env_radius_tau", 0.20),
            "pair_rbf_sigma": steering_args.get("ligand_env_pair_rbf_sigma", 0.18),

            # Less aggressive sharpening
            "contact_sharpness": steering_args.get("ligand_env_contact_sharpness", 1.5),
            "rbf_sharpness": steering_args.get("ligand_env_rbf_sharpness", 1.0),

            # Focus on strongest contacts only
            "topk_contacts": steering_args.get("ligand_env_topk_contacts", 4),
            "topk_pair_rbf": steering_args.get("ligand_env_topk_pair_rbf", 6),

            # Keep only the most useful parts first
            "include_shell_density": steering_args.get("ligand_env_include_shell_density", True),
            "include_shell_moments": steering_args.get("ligand_env_include_shell_moments", True),
            "include_protein_type_shell": steering_args.get("ligand_env_include_protein_type_shell", True),
            "include_pair_type_contact": steering_args.get("ligand_env_include_pair_type_contact", True),
            "include_pair_type_rbf": steering_args.get("ligand_env_include_pair_type_rbf", False),
            "include_ligtype_nearest": steering_args.get("ligand_env_include_ligtype_nearest", True),
        }
    )
)

    return potentials
