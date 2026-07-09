#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Minimal ensemble evaluation for Boltz2 CIF outputs.

Compares two folders of generated complex .cif files:
- baseline
- steered

Outputs:
1) contact probability maps and delta
2) simple mode clustering based on contact fingerprints
3) clash statistics

Recommended usage:
python eval_boltz_ensemble.py \
  --baseline_dir /path/to/baseline \
  --steered_dir /path/to/steered \
  --protein_chains A \
  --partner_chains B \
  --out_dir /path/to/out

Dependencies:
  pip install biopython numpy
"""

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Iterable, List

import numpy as np
from Bio.PDB import MMCIFParser


# -----------------------------
# basic helpers
# -----------------------------
def parse_chain_list(x: str) -> List[str]:
    return [i.strip() for i in x.split(",") if i.strip()]


def is_hydrogen(atom) -> bool:
    el = getattr(atom, "element", None)
    if el is not None and str(el).strip():
        return str(el).strip().upper() == "H"
    return atom.get_name().strip().upper().startswith("H")


def is_water_residue(residue) -> bool:
    resname = residue.get_resname().strip().upper()
    return resname in {"HOH", "WAT", "H2O"}


def residue_label(chain_id: str, residue) -> str:
    hetflag, seqid, icode = residue.id
    resname = residue.get_resname().strip()
    icode = icode.strip() if isinstance(icode, str) else ""
    # example: A:123:ALA or B:1:H_LIG:LIG
    if hetflag and hetflag.strip():
        return f"{chain_id}:{seqid}{icode}:{hetflag.strip()}:{resname}"
    return f"{chain_id}:{seqid}{icode}:{resname}"


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


# -----------------------------
# structure extraction
# -----------------------------
@dataclass
class ParsedStructure:
    protein_res_atoms: Dict[str, np.ndarray]   # label -> (n_atoms, 3)
    partner_res_atoms: Dict[str, np.ndarray]   # label -> (n_atoms, 3)
    protein_all_atoms: np.ndarray              # (N, 3)
    partner_all_atoms: np.ndarray              # (M, 3)


def parse_cif_structure(
    cif_path: Path,
    protein_chains: List[str],
    partner_chains: List[str],
    keep_hydrogens: bool = False,
) -> ParsedStructure:
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure(cif_path.stem, str(cif_path))
    model = next(structure.get_models())

    protein_res_atoms: Dict[str, List[np.ndarray]] = {}
    partner_res_atoms: Dict[str, List[np.ndarray]] = {}

    protein_all = []
    partner_all = []

    protein_chain_set = set(protein_chains)
    partner_chain_set = set(partner_chains)

    for chain in model:
        cid = chain.id
        if cid not in protein_chain_set and cid not in partner_chain_set:
            continue

        for residue in chain:
            if is_water_residue(residue):
                continue

            coords = []
            for atom in residue:
                if (not keep_hydrogens) and is_hydrogen(atom):
                    continue
                coords.append(atom.coord.astype(np.float32))

            if not coords:
                continue

            coords = np.asarray(coords, dtype=np.float32)
            label = residue_label(cid, residue)

            if cid in protein_chain_set:
                protein_res_atoms[label] = coords
                protein_all.append(coords)
            elif cid in partner_chain_set:
                partner_res_atoms[label] = coords
                partner_all.append(coords)

    protein_all_atoms = (
        np.concatenate(protein_all, axis=0) if len(protein_all) > 0 else np.zeros((0, 3), dtype=np.float32)
    )
    partner_all_atoms = (
        np.concatenate(partner_all, axis=0) if len(partner_all) > 0 else np.zeros((0, 3), dtype=np.float32)
    )

    return ParsedStructure(
        protein_res_atoms=protein_res_atoms,
        partner_res_atoms=partner_res_atoms,
        protein_all_atoms=protein_all_atoms,
        partner_all_atoms=partner_all_atoms,
    )


# -----------------------------
# contact / clash
# -----------------------------
def any_pair_below(a: np.ndarray, b: np.ndarray, cutoff: float) -> bool:
    if a.shape[0] == 0 or b.shape[0] == 0:
        return False
    cutoff2 = cutoff * cutoff
    # small residue-level arrays; direct broadcast is fine
    diff = a[:, None, :] - b[None, :, :]
    d2 = np.sum(diff * diff, axis=-1)
    return bool(np.any(d2 < cutoff2))


def count_pairs_below_and_dmin(
    a: np.ndarray,
    b: np.ndarray,
    cutoff: float,
    severe_cutoff: float,
    chunk_size: int = 512,
) -> Tuple[int, int, float]:
    """
    Counts protein-partner atom pairs below cutoff and severe_cutoff.
    Returns: (n_clash, n_severe, min_distance)
    """
    if a.shape[0] == 0 or b.shape[0] == 0:
        return 0, 0, float("inf")

    cutoff2 = cutoff * cutoff
    severe2 = severe_cutoff * severe_cutoff
    n_clash = 0
    n_severe = 0
    dmin2 = float("inf")

    for i in range(0, a.shape[0], chunk_size):
        aa = a[i : i + chunk_size]  # (c, 3)
        diff = aa[:, None, :] - b[None, :, :]
        d2 = np.sum(diff * diff, axis=-1)
        n_clash += int(np.count_nonzero(d2 < cutoff2))
        n_severe += int(np.count_nonzero(d2 < severe2))
        local_min = float(np.min(d2))
        if local_min < dmin2:
            dmin2 = local_min

    return n_clash, n_severe, math.sqrt(dmin2)


def contact_matrix_from_parsed(
    parsed: ParsedStructure,
    protein_axis: List[str],
    partner_axis: List[str],
    contact_cutoff: float,
) -> np.ndarray:
    """
    Binary residue-level contact matrix:
      shape = (n_protein_res, n_partner_res)
    entry=1 if any heavy-atom pair below cutoff.
    """
    mat = np.zeros((len(protein_axis), len(partner_axis)), dtype=np.uint8)

    for i, plabel in enumerate(protein_axis):
        if plabel not in parsed.protein_res_atoms:
            continue
        a = parsed.protein_res_atoms[plabel]

        for j, qlabel in enumerate(partner_axis):
            if qlabel not in parsed.partner_res_atoms:
                continue
            b = parsed.partner_res_atoms[qlabel]

            if any_pair_below(a, b, contact_cutoff):
                mat[i, j] = 1

    return mat


def flatten_contact(mat: np.ndarray) -> np.ndarray:
    return mat.reshape(-1).astype(np.uint8)


# -----------------------------
# clustering
# -----------------------------
def jaccard_distance_binary(a: np.ndarray, b: np.ndarray) -> float:
    inter = int(np.count_nonzero((a == 1) & (b == 1)))
    union = int(np.count_nonzero((a == 1) | (b == 1)))
    if union == 0:
        return 0.0
    return 1.0 - (inter / union)


def greedy_cluster_binary(
    fps: List[np.ndarray],
    dist_thresh: float = 0.30,
) -> Tuple[List[int], List[int]]:
    """
    Greedy clustering on binary fingerprints.
    Returns:
      labels: cluster index for each sample
      rep_indices: representative sample index per cluster
    """
    if len(fps) == 0:
        return [], []

    rep_indices: List[int] = [0]
    labels: List[int] = [0]

    for idx in range(1, len(fps)):
        x = fps[idx]
        dists = [jaccard_distance_binary(x, fps[r]) for r in rep_indices]
        best = int(np.argmin(dists))
        if dists[best] <= dist_thresh:
            labels.append(best)
        else:
            rep_indices.append(idx)
            labels.append(len(rep_indices) - 1)

    return labels, rep_indices


def occupancy_entropy(labels: List[int]) -> float:
    if len(labels) == 0:
        return 0.0
    vals, counts = np.unique(np.asarray(labels), return_counts=True)
    p = counts.astype(np.float64) / counts.sum()
    return float(-np.sum(p * np.log(p + 1e-12)))


def effective_cluster_count(labels: List[int], min_occ: float = 0.05) -> int:
    if len(labels) == 0:
        return 0
    vals, counts = np.unique(np.asarray(labels), return_counts=True)
    p = counts.astype(np.float64) / counts.sum()
    return int(np.sum(p >= min_occ))


# -----------------------------
# io helpers
# -----------------------------
def write_contact_csv(path: Path, protein_axis: List[str], partner_axis: List[str], mat: np.ndarray):
    """
    Writes long-form CSV:
      protein_residue, partner_residue, value
    """
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["protein_residue", "partner_residue", "value"])
        for i, plabel in enumerate(protein_axis):
            for j, qlabel in enumerate(partner_axis):
                w.writerow([plabel, qlabel, float(mat[i, j])])


def write_cluster_summary_csv(
    path: Path,
    file_names: List[str],
    labels: List[int],
    rep_indices: List[int],
):
    if len(labels) == 0:
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["cluster_id", "cluster_size", "occupancy", "representative_file"])
        return

    arr = np.asarray(labels)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["cluster_id", "cluster_size", "occupancy", "representative_file"])

        vals, counts = np.unique(arr, return_counts=True)
        total = len(labels)

        for cid, cnt in zip(vals, counts):
            rep_file = file_names[rep_indices[cid]] if cid < len(rep_indices) else ""
            w.writerow([int(cid), int(cnt), float(cnt / total), rep_file])


def write_sample_assignments_csv(path: Path, file_names: List[str], labels: List[int]):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file_name", "cluster_id"])
        for fn, lb in zip(file_names, labels):
            w.writerow([fn, int(lb)])


def write_clash_csv(
    path: Path,
    file_names: List[str],
    n_clash: List[int],
    n_severe: List[int],
    dmins: List[float],
):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file_name", "n_clash", "n_severe", "min_interface_distance"])
        for fn, c, s, d in zip(file_names, n_clash, n_severe, dmins):
            w.writerow([fn, int(c), int(s), float(d)])


# -----------------------------
# main evaluation logic
# -----------------------------
@dataclass
class EnsembleResult:
    file_names: List[str]
    contact_mats: List[np.ndarray]
    contact_prob: np.ndarray
    fingerprints: List[np.ndarray]
    cluster_labels: List[int]
    cluster_reps: List[int]
    entropy: float
    k_eff: int
    clash_counts: List[int]
    severe_counts: List[int]
    dmins: List[float]


def evaluate_folder(
    cif_dir: Path,
    protein_axis: List[str],
    partner_axis: List[str],
    protein_chains: List[str],
    partner_chains: List[str],
    contact_cutoff: float,
    clash_cutoff: float,
    severe_cutoff: float,
    cluster_dist_thresh: float,
    keep_hydrogens: bool = False,
) -> EnsembleResult:
    files = sorted(cif_dir.glob("*.cif"))
    if not files:
        raise FileNotFoundError(f"No .cif files found in: {cif_dir}")

    file_names = []
    contact_mats = []
    fingerprints = []
    clash_counts = []
    severe_counts = []
    dmins = []

    for fp in files:
        parsed = parse_cif_structure(
            fp,
            protein_chains=protein_chains,
            partner_chains=partner_chains,
            keep_hydrogens=keep_hydrogens,
        )

        cm = contact_matrix_from_parsed(
            parsed,
            protein_axis=protein_axis,
            partner_axis=partner_axis,
            contact_cutoff=contact_cutoff,
        )

        clash, severe, dmin = count_pairs_below_and_dmin(
            parsed.protein_all_atoms,
            parsed.partner_all_atoms,
            cutoff=clash_cutoff,
            severe_cutoff=severe_cutoff,
        )

        file_names.append(fp.name)
        contact_mats.append(cm)
        fingerprints.append(flatten_contact(cm))
        clash_counts.append(clash)
        severe_counts.append(severe)
        dmins.append(dmin)

    contact_prob = np.mean(np.stack(contact_mats, axis=0), axis=0)
    labels, reps = greedy_cluster_binary(fingerprints, dist_thresh=cluster_dist_thresh)
    ent = occupancy_entropy(labels)
    k_eff = effective_cluster_count(labels, min_occ=0.05)

    return EnsembleResult(
        file_names=file_names,
        contact_mats=contact_mats,
        contact_prob=contact_prob,
        fingerprints=fingerprints,
        cluster_labels=labels,
        cluster_reps=reps,
        entropy=ent,
        k_eff=k_eff,
        clash_counts=clash_counts,
        severe_counts=severe_counts,
        dmins=dmins,
    )


def collect_union_axes(
    files: List[Path],
    protein_chains: List[str],
    partner_chains: List[str],
    keep_hydrogens: bool = False,
) -> Tuple[List[str], List[str]]:
    protein_seen = {}
    partner_seen = {}

    for fp in files:
        parsed = parse_cif_structure(
            fp,
            protein_chains=protein_chains,
            partner_chains=partner_chains,
            keep_hydrogens=keep_hydrogens,
        )

        for k in parsed.protein_res_atoms.keys():
            protein_seen.setdefault(k, None)
        for k in parsed.partner_res_atoms.keys():
            partner_seen.setdefault(k, None)

    return list(protein_seen.keys()), list(partner_seen.keys())


# -----------------------------
# summary / comparison
# -----------------------------
# def summarize_contact_shift(
#     base_prob: np.ndarray,
#     steer_prob: np.ndarray,
# ) -> Dict[str, float]:
#     delta = steer_prob - base_prob
#     macs = float(np.mean(np.abs(delta)))  # mean absolute contact shift
#     mean_delta = float(np.mean(delta))
#     positive_frac = float(np.mean(delta > 0))
#     return {
#         "mean_absolute_contact_shift": macs,
#         "mean_delta_contact": mean_delta,
#         "fraction_entries_increased": positive_frac,
#     }


def _summarize_for_threshold(
    base_prob: np.ndarray,
    steer_prob: np.ndarray,
    threshold: float,
    topk: int = 50,
    eps: float = 1e-8,
) -> Dict[str, float]:
    mask_base = base_prob > threshold
    mask_steer = steer_prob > threshold
    mask_union = mask_base | mask_steer

    n_total = base_prob.size
    n_base = int(np.count_nonzero(mask_base))
    n_steer = int(np.count_nonzero(mask_steer))
    n_union = int(np.count_nonzero(mask_union))
    n_inter = int(np.count_nonzero(mask_base & mask_steer))

    # empty active region
    if n_union == 0:
        return {
            "active_region_ratio": 0.0,
            "mean_absolute_contact_shift_active": 0.0,
            "mean_delta_contact_active": 0.0,
            "fraction_entries_increased_active": 0.0,
            "fraction_entries_decreased_active": 0.0,
            "mean_abs_relative_shift_active": 0.0,
            "contact_map_jaccard_similarity": 0.0,
            "precision_steer_vs_base": 0.0,
            "recall_steer_vs_base": 0.0,
            "f1_steer_vs_base": 0.0,
            "new_contact_emergence_rate_union": 0.0,
            "new_contact_emergence_rate_steer": 0.0,
            "lost_contact_rate_union": 0.0,
            "lost_contact_rate_base": 0.0,
            "topk_overlap_ratio": 0.0,
        }

    delta = steer_prob - base_prob
    delta_active = delta[mask_union]
    base_active = base_prob[mask_union]
    

    macs_active = float(np.mean(np.abs(delta_active)))
    mean_delta_active = float(np.mean(delta_active))
    pos_frac_active = float(np.mean(delta_active > 0))
    neg_frac_active = float(np.mean(delta_active < 0))
    mean_abs_rel_shift = float(np.mean(np.abs(delta_active) / (base_active + eps)))

    new_contacts = (~mask_base) & mask_steer
    lost_contacts = mask_base & (~mask_steer)
    n_new = int(np.count_nonzero(new_contacts))
    n_lost = int(np.count_nonzero(lost_contacts))

    emergence_union = float(n_new / n_union)
    loss_union = float(n_lost / n_union)
    emergence_steer = float(n_new / max(n_steer, 1))
    loss_base = float(n_lost / max(n_base, 1))

    jaccard = float(n_inter / n_union)
    precision = float(n_inter / max(n_steer, 1))
    recall = float(n_inter / max(n_base, 1))
    f1 = float(2 * precision * recall / max((precision + recall), eps))

    # top-k overlap (optional, still informative at different thresholds)
    flat_base = base_prob.reshape(-1)
    flat_steer = steer_prob.reshape(-1)
    k = int(min(topk, flat_base.size))
    if k <= 0:
        topk_overlap = 0.0
    else:
        idx_b = np.argpartition(flat_base, -k)[-k:]
        idx_s = np.argpartition(flat_steer, -k)[-k:]
        topk_overlap = float(len(set(idx_b.tolist()) & set(idx_s.tolist())) / k)

    return {
        "active_region_ratio": float(n_union / n_total),
        "mean_absolute_contact_shift_active": macs_active,
        "mean_delta_contact_active": mean_delta_active,
        "fraction_entries_increased_active": pos_frac_active,
        "fraction_entries_decreased_active": neg_frac_active,
        "mean_abs_relative_shift_active": mean_abs_rel_shift,
        "contact_map_jaccard_similarity": jaccard,
        "precision_steer_vs_base": precision,
        "recall_steer_vs_base": recall,
        "f1_steer_vs_base": f1,
        "new_contact_emergence_rate_union": emergence_union,
        "new_contact_emergence_rate_steer": emergence_steer,
        "lost_contact_rate_union": loss_union,
        "lost_contact_rate_base": loss_base,
        "topk_overlap_ratio": topk_overlap,
    }


def summarize_contact_shift_multi_threshold(
    base_prob: np.ndarray,
    steer_prob: np.ndarray,
    thresholds: List[float] = [0.1, 0.3, 0.5],
    topk: int = 50,
) -> Dict[str, float]:
    """
    Run the same active-mask contact-shift summary for multiple thresholds.
    Output keys are suffixed by e.g. _thr0p1, _thr0p3, _thr0p5.
    """
    if base_prob.shape != steer_prob.shape:
        raise ValueError(f"Shape mismatch: {base_prob.shape} vs {steer_prob.shape}")

    out: Dict[str, float] = {}
    for thr in thresholds:
        stats = _summarize_for_threshold(base_prob, steer_prob, threshold=float(thr), topk=topk)
        suffix = f"_thr{str(thr).replace('.', 'p')}"
        for k, v in stats.items():
            out[k + suffix] = v
    return out


def interpret_results_multi_threshold_both(
    metrics: Dict[str, float],
    thresholds: Iterable[float] = (0.1, 0.3, 0.5),
    jaccard_stable: float = 0.8,
    jaccard_partial: float = 0.4,
    macs_warn: float = 0.30,
    emerg_warn: float = 0.20,
    loss_warn: float = 0.20,
) -> Tuple[str, Dict]:
    """
    Return:
      text: human-readable multi-line summary
      struct: JSON-friendly structured dict
    Expects keys suffixed like _thr0p1/_thr0p3/_thr0p5 from summarize_contact_shift_multi_threshold().
    """

    def suf(thr: float) -> str:
        return f"_thr{str(thr).replace('.', 'p')}"

    def get(key: str, thr: float, default=0.0) -> float:
        return float(metrics.get(key + suf(thr), default))

    thresholds = tuple(float(t) for t in thresholds)

    # choose overall reference threshold (prefer 0.3 if has active region)
    thr_overall = 0.3 if (0.3 in thresholds and get("active_region_ratio", 0.3, 0.0) > 0) else thresholds[0]
    if get("active_region_ratio", thr_overall, 0.0) == 0.0:
        # fallback to first threshold that has active region
        for t in thresholds:
            if get("active_region_ratio", t, 0.0) > 0:
                thr_overall = t
                break

    # ---- build STRUCT (JSON-friendly) ----
    struct = {
        "overall": {},
        "by_threshold": [],
        "schema": {
            "threshold_suffix": "_thr{thr with '.'->'p'}",
            "notes": "All rates are fractions in [0,1]."
        }
    }

    j_overall = get("contact_map_jaccard_similarity", thr_overall, 0.0)
    if j_overall > jaccard_stable:
        verdict = "stable"
    elif j_overall > jaccard_partial:
        verdict = "partial_rearrangement"
    else:
        verdict = "mode_switching"

    struct["overall"] = {
        "ref_threshold": thr_overall,
        "verdict": verdict,
        "active_region_ratio": get("active_region_ratio", thr_overall, 0.0),
        "jaccard": j_overall,
        "f1": get("f1_steer_vs_base", thr_overall, 0.0),
        "topk_overlap": get("topk_overlap_ratio", thr_overall, 0.0),
        "shift": {
            "macs": get("mean_absolute_contact_shift_active", thr_overall, 0.0),
            "mean_delta": get("mean_delta_contact_active", thr_overall, 0.0),
            "fraction_increased": get("fraction_entries_increased_active", thr_overall, 0.0),
            "fraction_decreased": get("fraction_entries_decreased_active", thr_overall, 0.0),
        },
        "dynamics": {
            "new_union": get("new_contact_emergence_rate_union", thr_overall, 0.0),
            "new_steer": get("new_contact_emergence_rate_steer", thr_overall, 0.0),
            "lost_union": get("lost_contact_rate_union", thr_overall, 0.0),
            "lost_base": get("lost_contact_rate_base", thr_overall, 0.0),
        }
    }

    for thr in thresholds:
        struct["by_threshold"].append({
            "threshold": thr,
            "active_region_ratio": get("active_region_ratio", thr, 0.0),
            "jaccard": get("contact_map_jaccard_similarity", thr, 0.0),
            "f1": get("f1_steer_vs_base", thr, 0.0),
            "topk_overlap": get("topk_overlap_ratio", thr, 0.0),
            "shift": {
                "macs": get("mean_absolute_contact_shift_active", thr, 0.0),
                "mean_delta": get("mean_delta_contact_active", thr, 0.0),
                "fraction_increased": get("fraction_entries_increased_active", thr, 0.0),
                "fraction_decreased": get("fraction_entries_decreased_active", thr, 0.0),
            },
            "dynamics": {
                "new_union": get("new_contact_emergence_rate_union", thr, 0.0),
                "new_steer": get("new_contact_emergence_rate_steer", thr, 0.0),
                "lost_union": get("lost_contact_rate_union", thr, 0.0),
                "lost_base": get("lost_contact_rate_base", thr, 0.0),
            }
        })

    # ---- build TEXT (human-readable) ----
    lines = []
    lines.append("=== Contact-shift interpretation (multi-threshold) ===")

    # overall line
    tag = {"stable": "🟢", "partial_rearrangement": "🟡", "mode_switching": "🔴"}[verdict]
    msg = {
        "stable": "Overall: binding/contact mode largely preserved",
        "partial_rearrangement": "Overall: partial rearrangement of contact mode",
        "mode_switching": "Overall: strong mode switching (contact topology changed a lot)",
    }[verdict]

    macs_o = struct["overall"]["shift"]["macs"]
    new_o = struct["overall"]["dynamics"]["new_union"]
    lost_o = struct["overall"]["dynamics"]["lost_union"]

    flags = []
    if macs_o > macs_warn:
        flags.append(f"high shift (MACS={macs_o:.2f})")
    if new_o > emerg_warn:
        flags.append(f"many new contacts (new={new_o:.2%})")
    if lost_o > loss_warn:
        flags.append(f"many lost contacts (lost={lost_o:.2%})")

    if flags:
        lines.append(f"{tag} {msg} | " + ", ".join(flags) + f"  [ref thr={thr_overall}]")
    else:
        lines.append(f"{tag} {msg}  [ref thr={thr_overall}]")

    lines.append("")

    # per threshold details
    for thr in thresholds:
        ar = get("active_region_ratio", thr, 0.0)
        if ar == 0.0:
            lines.append(f"[thr={thr}] active region: 0 (no contacts exceed threshold).")
            continue

        j = get("contact_map_jaccard_similarity", thr, 0.0)
        f1 = get("f1_steer_vs_base", thr, 0.0)
        topk = get("topk_overlap_ratio", thr, 0.0)

        if j > jaccard_stable:
            mode_txt = "🟢 stable"
        elif j > jaccard_partial:
            mode_txt = "🟡 partial rearrangement"
        else:
            mode_txt = "🔴 mode switching"

        macs = get("mean_absolute_contact_shift_active", thr, 0.0)
        md = get("mean_delta_contact_active", thr, 0.0)
        pos = get("fraction_entries_increased_active", thr, 0.0)
        neg = get("fraction_entries_decreased_active", thr, 0.0)

        emerg_u = get("new_contact_emergence_rate_union", thr, 0.0)
        emerg_s = get("new_contact_emergence_rate_steer", thr, 0.0)
        loss_u = get("lost_contact_rate_union", thr, 0.0)
        loss_b = get("lost_contact_rate_base", thr, 0.0)

        sev_bits = []
        if macs > macs_warn:
            sev_bits.append(f"MACS↑{macs:.2f}")
        if emerg_u > emerg_warn:
            sev_bits.append(f"new↑{emerg_u:.1%}")
        if loss_u > loss_warn:
            sev_bits.append(f"lost↑{loss_u:.1%}")

        sev_txt = (" | " + ", ".join(sev_bits)) if sev_bits else ""

        lines.append(
            f"[thr={thr}] active={ar:.2%} | Jaccard={j:.2f} ({mode_txt}) | "
            f"F1={f1:.2f} | topK-overlap={topk:.2f}{sev_txt}"
        )
        lines.append(f"        shift: MACS={macs:.2f}, meanΔ={md:+.2f}, increased={pos:.1%}, decreased={neg:.1%}")
        lines.append(
            f"        dynamics: new={emerg_u:.1%} (union), {emerg_s:.1%} (steer-active) | "
            f"lost={loss_u:.1%} (union), {loss_b:.1%} (base-active)"
        )

    text = "\n".join(lines)
    return text, struct


def summarize_clash_change(
    base: EnsembleResult,
    steer: EnsembleResult,
) -> Dict[str, float]:
    return {
        "baseline_mean_n_clash": float(np.mean(base.clash_counts)),
        "steered_mean_n_clash": float(np.mean(steer.clash_counts)),
        "baseline_mean_n_severe": float(np.mean(base.severe_counts)),
        "steered_mean_n_severe": float(np.mean(steer.severe_counts)),
        "baseline_mean_dmin": float(np.mean(base.dmins)),
        "steered_mean_dmin": float(np.mean(steer.dmins)),
        "baseline_frac_any_clash": float(np.mean(np.asarray(base.clash_counts) > 0)),
        "steered_frac_any_clash": float(np.mean(np.asarray(steer.clash_counts) > 0)),
    }


def summarize_modes(
    base: EnsembleResult,
    steer: EnsembleResult,
) -> Dict[str, float]:
    return {
        "baseline_k_eff": int(base.k_eff),
        "steered_k_eff": int(steer.k_eff),
        "baseline_entropy": float(base.entropy),
        "steered_entropy": float(steer.entropy),
        "k_eff_increased": bool(steer.k_eff > base.k_eff),
        "entropy_increased": bool(steer.entropy > base.entropy),
    }


# -----------------------------
# cli
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline_dir", required=True, type=str)
    ap.add_argument("--steered_dir", required=True, type=str)
    ap.add_argument("--protein_chains", required=True, type=str, help="comma-separated chain IDs")
    ap.add_argument("--partner_chains", required=True, type=str, help="comma-separated chain IDs")
    ap.add_argument("--out_dir", required=True, type=str)

    ap.add_argument("--contact_cutoff", type=float, default=4.5)
    ap.add_argument("--clash_cutoff", type=float, default=2.0)
    ap.add_argument("--severe_cutoff", type=float, default=1.5)
    ap.add_argument("--cluster_dist_thresh", type=float, default=0.30,
                    help="Jaccard distance threshold for greedy mode clustering")
    ap.add_argument("--keep_hydrogens", action="store_true", default=False)

    args = ap.parse_args()

    baseline_dir = Path(args.baseline_dir)
    steered_dir = Path(args.steered_dir)
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    protein_chains = parse_chain_list(args.protein_chains)
    partner_chains = parse_chain_list(args.partner_chains)

    baseline_files = sorted(baseline_dir.glob("*.cif"))
    steered_files = sorted(steered_dir.glob("*.cif"))
    all_files = baseline_files + steered_files

    if not baseline_files:
        raise FileNotFoundError(f"No .cif files in baseline_dir: {baseline_dir}")
    if not steered_files:
        raise FileNotFoundError(f"No .cif files in steered_dir: {steered_dir}")

    # build a common residue axis across both folders
    protein_axis, partner_axis = collect_union_axes(
        all_files,
        protein_chains=protein_chains,
        partner_chains=partner_chains,
        keep_hydrogens=args.keep_hydrogens,
    )

    if len(protein_axis) == 0:
        raise RuntimeError("No protein residues found for the specified protein_chains.")
    if len(partner_axis) == 0:
        raise RuntimeError("No partner residues found for the specified partner_chains.")

    # evaluate
    base = evaluate_folder(
        cif_dir=baseline_dir,
        protein_axis=protein_axis,
        partner_axis=partner_axis,
        protein_chains=protein_chains,
        partner_chains=partner_chains,
        contact_cutoff=args.contact_cutoff,
        clash_cutoff=args.clash_cutoff,
        severe_cutoff=args.severe_cutoff,
        cluster_dist_thresh=args.cluster_dist_thresh,
        keep_hydrogens=args.keep_hydrogens,
    )

    steer = evaluate_folder(
        cif_dir=steered_dir,
        protein_axis=protein_axis,
        partner_axis=partner_axis,
        protein_chains=protein_chains,
        partner_chains=partner_chains,
        contact_cutoff=args.contact_cutoff,
        clash_cutoff=args.clash_cutoff,
        severe_cutoff=args.severe_cutoff,
        cluster_dist_thresh=args.cluster_dist_thresh,
        keep_hydrogens=args.keep_hydrogens,
    )

    # write contact probabilities
    write_contact_csv(out_dir / "contact_prob_baseline.csv", protein_axis, partner_axis, base.contact_prob)
    write_contact_csv(out_dir / "contact_prob_steered.csv", protein_axis, partner_axis, steer.contact_prob)
    delta = steer.contact_prob - base.contact_prob
    write_contact_csv(out_dir / "delta_contact.csv", protein_axis, partner_axis, delta)

    # write cluster summaries
    write_cluster_summary_csv(
        out_dir / "cluster_summary_baseline.csv",
        base.file_names, base.cluster_labels, base.cluster_reps
    )
    write_cluster_summary_csv(
        out_dir / "cluster_summary_steered.csv",
        steer.file_names, steer.cluster_labels, steer.cluster_reps
    )
    write_sample_assignments_csv(
        out_dir / "sample_assignments_baseline.csv",
        base.file_names, base.cluster_labels
    )
    write_sample_assignments_csv(
        out_dir / "sample_assignments_steered.csv",
        steer.file_names, steer.cluster_labels
    )

    # write clash summaries
    write_clash_csv(
        out_dir / "clash_summary_baseline.csv",
        base.file_names, base.clash_counts, base.severe_counts, base.dmins
    )
    write_clash_csv(
        out_dir / "clash_summary_steered.csv",
        steer.file_names, steer.clash_counts, steer.severe_counts, steer.dmins
    )

    # write overall comparison summary
    summary = {
        "n_baseline_structures": len(base.file_names),
        "n_steered_structures": len(steer.file_names),
        "n_protein_residues_axis": len(protein_axis),
        "n_partner_residues_axis": len(partner_axis),
        # "contact_shift": summarize_contact_shift(base.contact_prob, steer.contact_prob),
        "contact_shift": summarize_contact_shift_multi_threshold(
            base.contact_prob, steer.contact_prob, thresholds=[0.1, 0.3, 0.5], topk=20), 
        "mode_summary": summarize_modes(base, steer),
        "clash_summary": summarize_clash_change(base, steer),
    }
    
    contact_text, contact_struct = interpret_results_multi_threshold_both(summary["contact_shift"])

    summary["contact_shift_interpretation"] = contact_struct

    (out_dir / "contact_shift_interpretation.txt").write_text(contact_text, encoding="utf-8")

    print(contact_text)

    with (out_dir / "comparison_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # print quick human-readable verdict
    print("\n=== Ensemble comparison summary ===")
    print(json.dumps(summary, indent=2))
    print("\nKey files written to:", out_dir)
    print("  - contact_prob_baseline.csv")
    print("  - contact_prob_steered.csv")
    print("  - delta_contact.csv")
    print("  - cluster_summary_baseline.csv")
    print("  - cluster_summary_steered.csv")
    print("  - clash_summary_baseline.csv")
    print("  - clash_summary_steered.csv")
    print("  - comparison_summary.json")
   

if __name__ == "__main__":
    main()
    
    
# python eval_boltz_ensemble.py \
#   --baseline_dir /path/to/baseline_cifs \
#   --steered_dir /path/to/steered_cifs \
#   --protein_chains A \
#   --partner_chains B \
#   --out_dir /path/to/eval_out


# --protein_chains A,C --partner_chains B,D