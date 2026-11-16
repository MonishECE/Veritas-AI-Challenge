# UrbanRay unified decision engine — implementation and audit

## What changed

UrbanRay now has **one decision interface** to the **two unchanged, independently trained models**. No new model is advertised as a third, jointly trained model. The shared geometry-aware feature frame is constructed once per scene-hour/albedo scenario, then evaluated by the original regression and specialist hotspot models.

| Component | Implementation | Task |
|---|---|---|
| Radiation specialist | `outputs/geometry_ai_upgrade/regression_surrogate.joblib` | Estimate simulated absorbed shortwave radiation (W/m²) for the map and modeled roof heat differences |
| Reflection specialist | `outputs/geometry_ai_upgrade/hotspot_surrogate.joblib` | Estimate and rank high-reflection walls; power the new specialist map layer |
| Unified engine | `src/decision_engine.py` | Validate model identity and matching data provenance, reuse inputs, route outputs, show model disagreement, and **abstain from approval** |
| Research screening | `screen_buildings()` | Join regression roof-solar-savings estimates with local wall changes predicted by the hotspot specialist under the **all-roofs** treatment scenarios |
| Offline dashboard | `outputs/unified_decision_engine/dashboard/UrbanRay_AI_Explorer_OFFLINE.html` | Preserve original dark-blue theme/maps, add unified decision panel and specialist hotspot map mode |
| New runtime mode | `run_physics_plus.py --radiation-mode compare_decision` | Run full physics vs routed regression output vs no-reflection using matched inputs, and include separate specialist/model-disagreement diagnostics |

### Preserved trained model fingerprints (SHA-256)

- Regression: `0ff3ae698bf8485e8c62e3b10bfc83f306bc912ab77346e9958fc53206a29960`
- Hotspot: `25046dea7c9079d667c17bb25a034a7b450831a84f98a9b35ef8f1b6299ba9df`

These hashes match the supplied **UrbanRay_AI_Enhanced** source directory exactly. Original dashboard assets remain in their prior locations as well.

## How the decisions work

1. Validate *both* trained models' role, prediction target, geometry-aware feature schema and source-data hash. Reject incompatible assets.
2. Calculate input features once, then run both models (with the existing nonnegativity/night/source-cap rules).
3. Route regression output to the main radiation map and roof absorbed-solar-heat comparisons; route hotspot output to wall-patch reflection maps and top-20 list.
4. Flag a face-hour when the model predictions differ by more than **15 W/m²** (an explicit **engineering review heuristic**, not a statistical interval). A source-cap intervention also forces a review flag for that entire batch.
5. For each building, calculate the modeled daily roof absorbed-solar-energy change under the **all roofs dark** versus **all roofs reflective** scenarios. For nearby *other-building* wall faces within **75 m**, separately report the hotspot-specialist's predicted additional reflected wall energy under those **same whole-neighborhood scenarios**. These surrounding-wall values are deliberately **not attributable to one building's roof**.
6. Return a *research-screening flag*, not a retrofit recommendation. Top-quartile roof benefit is marked as a priority for further study **while retaining nearby-facade review flags**; model disagreement, missing nearby coverage, and predicted wall increases remain visible.

**Important causal limitation:** Building X's local wall change in the table is a neighborhood context diagnostic under *all* roofs being changed; it is **not** the result of changing only Building X. Neighbor sets overlap and their kWh values must never be summed across buildings as independent effects. No costs, health effects, risk probabilities, or measured cooling are inferred.

**Important uncertainty limitation:** Disagreement between two ML models trained on the same simulator is not calibrated uncertainty, and small disagreement does not establish correctness.

## Evidence and results

Historical comparison: 2025-06-23, 20,976 correlated, held-out **simulator** face-hours from the previously inspected spatial/date partition. **Not an untouched new lockbox.**

| Existing model | Simulator MAE (W/m²) | Captured simulated top-20 wall-reflection exposure |
|---|---:|---:|
| Earlier released reflection-residual AI | 3.635 | 81.14% |
| Geometry-aware regression (retained) | 3.081 | 79.93% |
| Dedicated hotspot specialist (retained) | 3.022 | 85.88% |
| No-reflection rule | 7.898 | 11.46% |

The new engine routes the **same** regression and hotspot outputs; it does **not** establish a new accuracy improvement beyond those existing benchmarks. The mapping/screening run covers the full previously simulated June 23 day, both scenarios, 88 buildings, and 2,082 triangular patches, which is not the same partition as the 20,976-row held-out comparison.

The merged screening output has **22 'PRIORITY · CHECK FACADES'** and **66 'REVIEW NEIGHBOR FACADES'** cases under the explicit heuristic. Every building still requires field verification. These counts do not constitute success or fail rates.

### Tested

- **59 automated Python tests passed** (`python -m pytest -q tests`). Includes 8 added tests of role validation, model identity preservation, nonphysical-domain rejection, night zeroing, complete matched scenarios, non-causal neighboring-wall reporting, specialist selection, and dashboard data.
- **Chromium browser check passed** on the fully embedded HTML with no JavaScript errors. Tested changing the selected building, specialist map mode, populated hotspot table and visible decision panel. Screenshot: `outputs/unified_decision_engine/dashboard/UrbanRay_Unified_Decision_Engine_Preview.png`.
- **Synthetic-weather two-hour smoke test** completed for `compare_decision` (4 scene/scenario points per method, 12 summary rows; full face-hour records kept). This tests wiring, **not** real weather performance or speed advantage.
- Original model files checked byte-for-byte against the prior supplied folder.
- Package snapshot integrity can be checked with `python verify_checksums.py`.

### What remains unproven

- Actual roof/wall temperature or field solar reflection accuracy.
- Cross-city, seasonal or independent lockbox generalization.
- A speed advantage over the optimized sparse physics solver (the prior smoke benchmark found the AI slower).
- A causal impact from changing any *single* roof; the current scenarios alter all roofs together.
- **Development timeline:** We built the UrbanRay models and unified decision interface during November 14–15, 2026. See our repository history and contribution records.

## Open the app

Extract the archive, then double-click:

`outputs/unified_decision_engine/dashboard/UrbanRay_AI_Explorer_OFFLINE.html`

The dashboard embeds Plotly and data and works offline with a modern browser. Select any building in the ranking table, inspect its decision panel, and set **What to visualize → Hotspot specialist · modeled reflections** to display the second model's own map.

## Curated archive reproducibility boundary

The provided offline HTML, trained model files, archived benchmarks and test suite are ready to inspect. A full dashboard **rebuild** requires `outputs/historical_ml/physics_training_faces.csv` from the original, untrimmed 131 MB historical archive; that input is intentionally absent from this distribution. The 2025 five-date satellite aggregate likewise lacks the original per-date validator outputs. We do **not** claim those full computations were re-executed from this ZIP. Use `python competition_evidence_audit.py`, `python -m pytest -q tests`, and `python verify_checksums.py` for reproducible checks over the files that are present.

## Re-run the project

```bash
python -m pip install -r requirements.txt
python -m pytest -q tests
python build_ai_dashboard.py \
  --model outputs/geometry_ai_upgrade/regression_surrogate.joblib \
  --hotspot-model outputs/geometry_ai_upgrade/hotspot_surrogate.joblib \
  --heldout outputs/geometry_ai_upgrade/retrospective_heldout_predictions_SIMULATED.csv \
  --output outputs/unified_decision_engine/dashboard
python verify_checksums.py
```

The checksum verification above applies to the packaged **unaltered snapshot**; a deliberate rebuild will naturally change time-dependent audit metadata and must be followed by intentional manifest regeneration if desired. Avoid blindly treating a post-rebuild checksum mismatch as evidence of corruption.

To exercise the runtime with **explicitly synthetic** two-hour smoke-weather inputs only:

```bash
python run_physics_plus.py \
  --weather-csv outputs/ai_upgrade/SMOKE_ONLY_daylight_2hour_weather.csv \
  --output outputs/unified_decision_engine/SMOKE_ONLY_2H_COMPARISON \
  --radiation-mode compare_decision \
  --rays 1 --shadow-samples 1 --warmup-hours 0 --hours 2 --bounces 2
```

The output `physics_training_faces.csv` identifies `decision_ai` and adds `hotspot_specialist_abs_Wm2`, `dual_model_disagreement_Wm2`, and `decision_review_required` on the engine rows. Do **not** use smoke data as independent observed evidence.
