import json
import csv
import re
from pathlib import Path
from typing import Any, Dict, List, Optional


PAIRWISE_ROOT = Path("/home/xukai_cluster/boltz_series/boltz/eval_out_anchor_pairwise")
OUTPUT_CSV = PAIRWISE_ROOT / "anchor_pairwise_benchmark.csv"
SUMMARY_NAME = "comparison_summary.json"


def safe_get(d: Dict[str, Any], *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def parse_pair_name(dirname: str):
  
    m = re.match(r"^(.+?)_(TO|VS)_(.+)$", dirname)
    if not m:
        return None, None, None
    anchor_a, mode, anchor_b = m.group(1), m.group(2), m.group(3)
    comparison_mode = "directed" if mode == "TO" else "undirected"
    return anchor_a, anchor_b, comparison_mode


def extract_threshold_metrics(summary: Dict[str, Any], threshold_key: str = "thr_0p3") -> Dict[str, Any]:
    """
        summary["thresholds"]["thr_0p3"]
      summary["by_threshold"]["thr_0p3"]
    """
    thr_block = (
        safe_get(summary, "thresholds", threshold_key)
        or safe_get(summary, "by_threshold", threshold_key)
        or {}
    )

    return {
        "threshold_key": threshold_key,
        "verdict": safe_get(thr_block, "verdict"),
        "ref_threshold": safe_get(thr_block, "ref_threshold"),
        "jaccard": safe_get(thr_block, "jaccard"),
        "f1": safe_get(thr_block, "f1"),
        "topk_overlap": safe_get(thr_block, "topk_overlap"),
        "fraction_entries_increased_active": safe_get(thr_block, "fraction_entries_increased_active"),
        "fraction_entries_decreased_active": safe_get(thr_block, "fraction_entries_decreased_active"),
        "new_contact_emergence_rate_steer": safe_get(thr_block, "new_contact_emergence_rate_steer"),
        "lost_contact_rate_base": safe_get(thr_block, "lost_contact_rate_base"),
    }


def extract_threshold_metrics_fallback(summary: Dict[str, Any], threshold_key: str = "thr_0p3") -> Dict[str, Any]:
   
    overall = safe_get(summary, "overall", default={}) or {}
    interpretations = safe_get(summary, "interpretations", default={}) or {}

    return {
        "threshold_key": threshold_key,
        "verdict": safe_get(interpretations, "overall", "verdict"),
        "ref_threshold": safe_get(
            interpretations, "overall", "ref_threshold",
            default=safe_get(overall, "ref_threshold")
        ),
        "jaccard": safe_get(overall, "jaccard"),
        "f1": safe_get(overall, "f1"),
        "topk_overlap": safe_get(overall, "topk_overlap"),
        "fraction_entries_increased_active": safe_get(overall, "fraction_entries_increased_active_thr0p3"),
        "fraction_entries_decreased_active": safe_get(overall, "fraction_entries_decreased_active_thr0p3"),
        "new_contact_emergence_rate_steer": safe_get(overall, "new_contact_emergence_rate_steer_thr0p3"),
        "lost_contact_rate_base": safe_get(overall, "lost_contact_rate_base_thr0p3"),
    }


def extract_row(out_dir: Path, summary_path: Path) -> Dict[str, Any]:
    with open(summary_path, "r") as f:
        summary = json.load(f)

    anchor_a, anchor_b, comparison_mode = parse_pair_name(out_dir.name)
    if anchor_a is None:
        raise ValueError(f"Cannot parse pair name from directory: {out_dir.name}")

    row = {
        "anchor_a": anchor_a,
        "anchor_b": anchor_b,
        "comparison_mode": comparison_mode,
        "pair_name": out_dir.name,
        "out_dir": str(out_dir),
        "summary_path": str(summary_path),

        "n_baseline_samples": safe_get(summary, "n_baseline_samples"),
        "n_steered_samples": safe_get(summary, "n_steered_samples"),

        "baseline_mean_n_clash": safe_get(summary, "baseline_mean_n_clash"),
        "steered_mean_n_clash": safe_get(summary, "steered_mean_n_clash"),
        "baseline_frac_any_clash": safe_get(summary, "baseline_frac_any_clash"),
        "steered_frac_any_clash": safe_get(summary, "steered_frac_any_clash"),
        "baseline_mean_dmin": safe_get(summary, "baseline_mean_dmin"),
        "steered_mean_dmin": safe_get(summary, "steered_mean_dmin"),

        "baseline_k_eff": safe_get(summary, "baseline_k_eff"),
        "steered_k_eff": safe_get(summary, "steered_k_eff"),
        "baseline_entropy": safe_get(summary, "baseline_entropy"),
        "steered_entropy": safe_get(summary, "steered_entropy"),
        "k_eff_increased": safe_get(summary, "k_eff_increased"),
        "entropy_increased": safe_get(summary, "entropy_increased"),
    }

    thr = extract_threshold_metrics(summary, threshold_key="thr_0p3")
    if thr["jaccard"] is None and thr["verdict"] is None:
        thr = extract_threshold_metrics_fallback(summary, threshold_key="thr_0p3")

    row.update({
        "threshold_key": thr["threshold_key"],
        "ref_threshold": thr["ref_threshold"],
        "verdict": thr["verdict"],
        "jaccard": thr["jaccard"],
        "f1": thr["f1"],
        "topk_overlap": thr["topk_overlap"],
        "fraction_entries_increased_active": thr["fraction_entries_increased_active"],
        "fraction_entries_decreased_active": thr["fraction_entries_decreased_active"],
        "new_contact_emergence_rate_steer": thr["new_contact_emergence_rate_steer"],
        "lost_contact_rate_base": thr["lost_contact_rate_base"],
    })

   
    bk = row["baseline_k_eff"]
    sk = row["steered_k_eff"]
    be = row["baseline_entropy"]
    se = row["steered_entropy"]
    bd = row["baseline_mean_dmin"]
    sd = row["steered_mean_dmin"]
    bc = row["baseline_mean_n_clash"]
    sc = row["steered_mean_n_clash"]

    row["delta_k_eff"] = (sk - bk) if (bk is not None and sk is not None) else None
    row["delta_entropy"] = (se - be) if (be is not None and se is not None) else None
    row["delta_mean_dmin"] = (sd - bd) if (bd is not None and sd is not None) else None
    row["delta_mean_n_clash"] = (sc - bc) if (bc is not None and sc is not None) else None

    return row


def collect_rows(pairwise_root: Path, summary_name: str) -> List[Dict[str, Any]]:
    rows = []
    for out_dir in sorted(pairwise_root.iterdir()):
        if not out_dir.is_dir():
            continue

        summary_path = out_dir / summary_name
        if not summary_path.exists():
            print(f"[skip] not found: {summary_path}")
            continue

        try:
            row = extract_row(out_dir, summary_path)
            rows.append(row)
            print(f"[ok] {out_dir.name}")
        except Exception as e:
            print(f"[error] {out_dir.name}: {e}")

    return rows


def sort_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    verdict_order = {
        "mode_switching": 0,
        "partial_shift": 1,
        "minor_shift": 2,
        "similar": 3,
        None: 9,
    }

    def key(r):
        return (
            verdict_order.get(r.get("verdict"), 8),
            -(r.get("jaccard") or -1),   
            -(r.get("f1") or -1),
            r.get("anchor_a") or "",
            r.get("anchor_b") or "",
        )

    return sorted(rows, key=key)


def write_csv(rows: List[Dict[str, Any]], output_csv: Path):
    if not rows:
        print("No rows collected. CSV not written.")
        return

    fieldnames = [
        "anchor_a",
        "anchor_b",
        "comparison_mode",
        "pair_name",
        "out_dir",
        "summary_path",

        "n_baseline_samples",
        "n_steered_samples",

        "baseline_mean_n_clash",
        "steered_mean_n_clash",
        "delta_mean_n_clash",
        "baseline_frac_any_clash",
        "steered_frac_any_clash",
        "baseline_mean_dmin",
        "steered_mean_dmin",
        "delta_mean_dmin",

        "baseline_k_eff",
        "steered_k_eff",
        "delta_k_eff",
        "baseline_entropy",
        "steered_entropy",
        "delta_entropy",
        "k_eff_increased",
        "entropy_increased",

        "threshold_key",
        "ref_threshold",
        "verdict",
        "jaccard",
        "f1",
        "topk_overlap",
        "fraction_entries_increased_active",
        "fraction_entries_decreased_active",
        "new_contact_emergence_rate_steer",
        "lost_contact_rate_base",
    ]

    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote CSV to: {output_csv}")


def main():
    rows = collect_rows(PAIRWISE_ROOT, SUMMARY_NAME)
    rows = sort_rows(rows)
    write_csv(rows, OUTPUT_CSV)


if __name__ == "__main__":
    main()