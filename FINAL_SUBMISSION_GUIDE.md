# UrbanRay Unified Decision Engine — final package guide

## What this ZIP contains

One **unified AI system** for modeling solar exposure and screening research priorities. Its original internal model assets are retained, not retrained or merged into a single weight matrix. Main interface: `outputs/unified_decision_engine/dashboard/UrbanRay_AI_Explorer_OFFLINE.html` (open in browser without a server). Code is in `src/`, with runner `run_physics_plus.py`, simulation inputs in `data/`, tests in `tests/`, and the original trained weight files in `outputs/geometry_ai_upgrade/`.

The package retains core reproducibility assets, key simulator-based benchmark reports, input Landsat CSVs, and `validate_unified_landsat.py`. It omits the 131 MB archived face-hour CSV, some redundant old dashboards and raw intermediate predictions. The original large archive can supply those if complete historical reconstruction is needed. `CHECKSUMS_SHA256.json` is the refreshed integrity manifest for this curated ZIP. Original source archives and repository history should be retained separately for provenance.

## Windows CMD quick start

Open this extracted folder in File Explorer; type `cmd` in the address bar.

```bat
py -m pip install -r requirements.txt
py -X utf8 -m pytest -q tests
start "" "outputs\unified_decision_engine\dashboard\UrbanRay_AI_Explorer_OFFLINE.html"
```

For an end-to-end smoke test using the packaged synthetic input (not observational evidence):

```bat
py run_physics_plus.py --weather-csv outputs/ai_upgrade/SMOKE_ONLY_daylight_2hour_weather.csv --output outputs/unified_decision_engine/NEW_SMOKE_TEST --radiation-mode compare_decision --rays 1 --shadow-samples 1 --warmup-hours 0 --hours 2 --bounces 2
```

For fresh observational checks, first fetch real historical weather, then simulate with `--radiation-mode compare_decision`, `--warmup-hours 48`, `--hours 24`, and run `validate_unified_landsat.py --faces [simulation output]/physics_training_faces.csv --pixels urbanray_qualified_landsat_pixels_2025_2026.csv --out [output folder] --max-st-uncertainty 5.0`. The validation script checks weather provenance.

## Five-date 2025 evidence (reported aggregate; raw per-date outputs unavailable)

`VALIDATION_2025_SUMMARY.json` transcribes the **five user-provided results**, not newly executed validations in this package. Mean matched-grid-cell MAE, equally weighted across five dates: Unified AI **6.227°C**, physics **6.169°C**, no-reflection **6.323°C**. These are exploratory comparisons against Landsat surface-temperature *proxies*, not field-measured roof accuracy. Three of five development dates favor Unified AI over no-reflection. Small differences must not be overstated, and wall-hotspot specialist accuracy is not validated by Landsat.

## Submission boundaries

Present **UrbanRay as one AI decision product**; its internal specialized computations are architecture, not competing entries. Keep provenance of synthetic versus observational data explicit. No evidence of measured cooling interventions, independently verified rooftop temperature accuracy, or calibrated uncertainty is included. The displayed `review` flags indicate research screening, not verified safety or installation advice.

See `UNIFIED_DECISION_ENGINE_REPORT.md` and `SCIENTIFIC_STATUS.md` for full details and limitations.

## Decision validation evidence (new, retrospective)

To audit actual candidate selections and guard against overstated observational claims, run `python decision_validation_gate.py` and open `outputs/decision_validation_gate/report.md`. This reruns archived roof and wall selection comparisons and scene-equal summary arithmetic without retraining. The package provides no new field observations; the evidence gate correctly reports that real-world decision superiority is **not established**. See `tests/test_decision_validation_gate.py`.
