"""Real-observation exploratory evaluation of UrbanRay's *unified* AI radiation path.

Place beside run_physics_plus.py and validate_landsat_pixels.py in the unified
project, then run with --faces, --pixels, --out. Calls the original footprint,
quality, Landsat, and thermal-proxy matching function unchanged. Does NOT
retrain, calibrate, or edit either model.

Important: Landsat LST vs the model's bolometric emission proxy is not direct
roof-temperature validation. The specialist hotspot model is tested for
integration, not independently against wall-reflection observations here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from validate_landsat_pixels import run as match_landsat

METHODS = ("physics", "decision_ai", "no_reflection")
KEYS = ("scene_id", "acquisition_utc", "center_lon", "center_lat")


def ensure_valid_audit(audit: dict) -> None:
    if audit.get("radiation_method") != "compare_decision":
        raise ValueError("Use --radiation-mode compare_decision, not compare")
    if set(audit.get("compared_methods", [])) != set(METHODS):
        raise ValueError("Run must contain physics, decision_ai, and no_reflection")
    if audit.get("weather_provenance_verified") is not True or "Open-Meteo" not in str(audit.get("weather_source", "")):
        raise ValueError("Real-weather data with verified Open-Meteo provenance is required")
    if audit.get("context_included"):
        raise ValueError("The supplied Landsat matcher only supports the core 88 buildings")
    if int(audit.get("warmup_hours", 0)) < 24 or int(audit.get("evaluated_hours", 0)) < 24:
        raise ValueError("Use at least 24 warm-up hours and 24 evaluation hours")
    if "SMOKE_ONLY" in str(audit.get("weather_file", "")).upper():
        raise ValueError("Cannot validate synthetic smoke-test weather as observations")
    if audit.get("decision_uses_two_original_unchanged_models") is not True:
        raise ValueError("Audit does not confirm the two unified-engine specialists were loaded")


def summarize_paired_pixels(pixels_by_method: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    merged = None
    for method in METHODS:
        d = pixels_by_method[method].copy()
        required = set(KEYS) | {"observed_landsat_lst_C", "modeled_emission_proxy_C", "landsat_uncertainty_K"}
        missing = required.difference(d.columns)
        if missing:
            raise ValueError(f"Missing matched-pixel columns for {method}: {sorted(missing)}")
        if d.duplicated(list(KEYS)).any():
            raise ValueError(f"Duplicate matched satellite cells for {method}")
        # Fail closed if any method has missing or nonphysical measurement metadata.
        for numerical in ("center_lon", "center_lat", "observed_landsat_lst_C",
                          "modeled_emission_proxy_C", "landsat_uncertainty_K"):
            vals = pd.to_numeric(d[numerical], errors="coerce").to_numpy(dtype=float)
            if not np.isfinite(vals).all():
                raise ValueError(f"Nonfinite or nonnumeric {numerical} for {method}")
        if (d["landsat_uncertainty_K"].to_numpy(dtype=float) < 0).any():
            raise ValueError(f"Negative satellite uncertainty for {method}")
        d = d[list(KEYS) + ["observed_landsat_lst_C", "modeled_emission_proxy_C", "landsat_uncertainty_K"]]
        d = d.rename(columns={"observed_landsat_lst_C": f"observed_{method}_C",
                              "modeled_emission_proxy_C": f"{method}_proxy_C",
                              "landsat_uncertainty_K": f"uncertainty_{method}_K"})
        merged = d if merged is None else merged.merge(d, on=list(KEYS), how="inner", validate="one_to_one")
    method_coverage = {method: {
        "input_grid_cells": int(len(pixels_by_method[method])),
        "paired_grid_cells": int(len(merged)),
        "unpaired_grid_cells_excluded": int(len(pixels_by_method[method]) - len(merged)),
    } for method in METHODS}
    if merged.empty:
        return merged, pd.DataFrame(), {"status": "NO_COMMON_SATELLITE_MATCHES", "paired_scenes": 0,
                                        "matched_grid_cells": 0, "roof_accuracy_established": False,
                                        "pairwise_coverage": method_coverage}
    for method in METHODS[1:]:
        if not np.allclose(merged[f"observed_{method}_C"].to_numpy(dtype=float),
                           merged["observed_physics_C"].to_numpy(dtype=float), atol=1e-8, rtol=0):
            raise ValueError("The paired models did not use identical Landsat observations")
        if not np.allclose(merged[f"uncertainty_{method}_K"].to_numpy(dtype=float),
                           merged["uncertainty_physics_K"].to_numpy(dtype=float), atol=1e-8, rtol=0):
            raise ValueError("The paired models used inconsistent Landsat uncertainty metadata")
    merged = merged.rename(columns={"observed_physics_C": "observed_landsat_C",
                                    "uncertainty_physics_K": "landsat_uncertainty_K"})
    merged = merged.drop(columns=[f"observed_{m}_C" for m in METHODS[1:]] +
                                 [f"uncertainty_{m}_K" for m in METHODS[1:]])
    for method in METHODS:
        merged[f"error_{method}_C"] = merged[f"{method}_proxy_C"] - merged["observed_landsat_C"]
    results = []
    for (scene_id, acquisition), group in merged.groupby(["scene_id", "acquisition_utc"], sort=True):
        r = {"scene_id": str(scene_id), "acquisition_utc": str(acquisition),
             "matched_grid_cells": int(len(group)),
             "observed_scene_mean_C": float(group.observed_landsat_C.mean()),
             "mean_landsat_uncertainty_K": float(group.landsat_uncertainty_K.mean())}
        for method in METHODS:
            diff = group[f"error_{method}_C"].to_numpy(dtype=float)
            r[f"{method}_scene_mean_bias_C"] = float(diff.mean())
            r[f"{method}_scene_mean_abs_error_C"] = float(abs(diff.mean()))
            r[f"{method}_grid_cell_MAE_C"] = float(abs(diff).mean())
        results.append(r)
    scenes = pd.DataFrame(results)
    aggregate = {}
    for method in METHODS:
        aggregate[method] = {
            "equal_scene_mean_abs_error_C": float(scenes[f"{method}_scene_mean_abs_error_C"].mean()),
            "equal_scene_mean_grid_cell_MAE_C": float(scenes[f"{method}_grid_cell_MAE_C"].mean()),
            "equal_scene_mean_bias_C": float(scenes[f"{method}_scene_mean_bias_C"].mean()),
        }
    return merged, scenes, {
        "status": "EXPLORATORY_PAIRED_UNIFIED_ENGINE_SATELLITE_PROXY_NOT_ROOF_VALIDATION",
        "paired_scenes": int(len(scenes)),
        "matched_grid_cells": int(len(merged)),
        "pairwise_coverage": method_coverage,
        "scene_results": aggregate,
        "decision_ai_beats_no_reflection_on_scene_mean_error":
            bool(aggregate["decision_ai"]["equal_scene_mean_abs_error_C"] < aggregate["no_reflection"]["equal_scene_mean_abs_error_C"]),
        "roof_accuracy_established": False,
        "hotspot_specialist_observational_accuracy_established": False,
        "limitations": [
            "Landsat surface-temperature retrieval is not a modeled roof temperature measurement",
            "The modeled bolometric emission proxy is not the Landsat spectral atmospheric inversion",
            "Matched 30 m grid cells have correlated native thermal information and are not independent roof observations",
            "Both AI models were trained on simulation, not observational field labels",
            "The specialist's wall-reflection outputs have no independent Landsat wall validation here",
            "The 5.0 K uncertainty threshold admits relatively uncertain satellite measurements",
            "Use 2025 development dates to diagnose; already-inspected 2026 dates are exploratory, not untouched holdouts",
        ],
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--faces", required=True, type=Path)
    p.add_argument("--pixels", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--max-st-uncertainty", type=float, default=5.0)
    args = p.parse_args()
    if not args.faces.is_file() or not args.pixels.is_file():
        raise FileNotFoundError("Missing physics face CSV or real Landsat pixels CSV")
    audit_path = args.faces.parent / "audit.json"
    if not audit_path.exists():
        raise FileNotFoundError("Missing audit.json next to the physics face CSV")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    ensure_valid_audit(audit)
    if not 0 < args.max_st_uncertainty <= 10:
        raise ValueError("Invalid --max-st-uncertainty; use the documented 5.0 K threshold")
    args.out.mkdir(parents=True, exist_ok=True)
    matches = {}
    method_counts = {}
    for method in METHODS:
        sub = args.out / f"{method}_EXPLORATORY"
        result = match_landsat(args.pixels, faces_csv=args.faces, out=sub,
                               method=method, max_st_uncertainty=args.max_st_uncertainty)
        matches[method] = pd.read_csv(sub / "matched_pixels_EXPLORATORY.csv")
        method_counts[method] = {"usable_scenes": result["usable_scene_count"],
                                 "matched_grid_cells": result["qualified_paired_30m_grid_cells"]}
        print(f"{method}: {method_counts[method]['usable_scenes']} usable scene(s), "
              f"{method_counts[method]['matched_grid_cells']} matched cells")
    paired, scene_summary, report = summarize_paired_pixels(matches)
    paired.to_csv(args.out / "paired_matched_grid_cells_EXPLORATORY.csv", index=False)
    scene_summary.to_csv(args.out / "paired_scene_metrics_EXPLORATORY.csv", index=False)
    report["max_landsat_st_uncertainty_k"] = args.max_st_uncertainty
    report["per_method_input_counts"] = method_counts
    report["faces_sha256"] = hashlib.sha256(args.faces.read_bytes()).hexdigest()
    report["satellite_file_sha256"] = hashlib.sha256(args.pixels.read_bytes()).hexdigest()
    report["simulation_weather_sha256"] = audit.get("weather_sha256")
    (args.out / "PAIRED_UNIFIED_VALIDATION_STATUS.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
