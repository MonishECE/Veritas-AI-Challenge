# Scientific evidence status — Veritas 2026 release

The main offline dashboard is a **unified, two-specialist model decision interface**. Against the previously inspected simulator-held-out date/spatial blocks, the geometry-aware regression model reports **3.08 W/m² MAE**, versus **7.90 W/m²** for no reflection; the specialist captures **85.88%** of top-20 modeled wall-reflection exposure versus **11.46%** for the direct-exposure rule. These are **simulation-referenced** results, not measurements of roof or wall temperature. Top-K rooftop portion selection did not improve over the baseline at K=3,5,10.

A five-date **2025 development** AI/physics/no-reflection comparison against Landsat surface-temperature retrievals has been **reported** in `VALIDATION_2025_SUMMARY.json` (135 correlated matched grid-cell rows, mean grid-cell MAE **6.23/6.17/6.32 °C**). AI is better than no-reflection on three of five dates and worse than physics on all five. Original per-date raw face-hour outputs and matching logs are **not included** in this curated archive, so this is a reported exploratory result, not a new independently replicated validation or proof of individual roof-temperature accuracy. The separately archived 2026 Landsat panel is **physics-only**.

`COMPETITION_EVIDENCE_AUDIT.md` checks the packaged metrics and evidence provenance. No observed cooling intervention, cross-city holdout, calibrated uncertainty, validated speedup, or causal single-roof effect is demonstrated.

---

## New decision task — 15 Nov 2026

The AI reflection-residual model selects high-reflection wall hotspots **against simulator reference labels** more effectively than a direct-exposure heuristic (top-20: 81.14% vs 11.46% of the simulator top-20 benchmark). On 3/5/10 held-out roof-portion selections it provides no decision improvement over no-reflection. The archived 2026 Landsat comparison is **physics-only** and exploratory; it cannot be described as AI or measured cool-roof validation. The archived 2026 physics-only Landsat panel must not be presented as validation of the AI. A separate five-date 2025 AI proxy summary is described above and remains unreproduced in this ZIP. See `REAL_WORLD_UPGRADE_REPORT.md`.

---

## AI reflection-correction upgrade (15 November 2026)

- New AI model **learns the extra simulated shortwave absorption caused by reflections** rather than an observed roof temperature.
- Strict date + spatial-block held-out error: **3.635 W/m² MAE** vs **7.898 W/m²** no-reflection baseline. Both are relative to multibounce **simulated** labels.
- Building-level dashboard rankings are based on a June 23, 2025 physics-simulation day with assumed dark/reflective roof albedos. Rank correlation with simulator is not independent environmental validation.
- Three radiation codepaths now drive independent thermal states; two-hour synthetic-only smoke test verified execution. The reported five-date 2025 paired proxy comparison is described above; raw per-date runs are not reproduced here.
- Observed AI inference was **slower** than the sparse shortwave operator in the smoke benchmark. Do not claim accelerated simulation.
- Full methodology and audit: `AI_UPGRADE_REPORT.md`.

---

# Scientific status — latest available evidence

**Observed-data comparison performed, but real-world roof-temperature accuracy remains unproven.**

- **Physics input:** 88 official NYC building records, simplified extruded meshes; material properties and some thermal boundary conditions are assumptions.
- **Simulator ML:** trained and tested against physically simulated shortwave absorption, not independently observed irradiance or roof temperature; `outputs/ml_holdout/` is the complete current simulator benchmark.
- **Independent satellite inputs:** the three latest Landsat 2025–2026 CSV exports at package root; 150 candidate scenes, 44 shortlisted scenes, 1,104 qualified resampled-grid-cell rows.
- **Exploratory comparison:** 10 acquisition scenes in `outputs/multidate_validation/scene_level_EXPLORATORY.csv`; five 2025 development, five 2026 subsequent evaluation scenes. 272 gridded (correlated) matches across scenes.
- **Held-out (2026) scene summary:** raw 3.526 °C MAE / 4.061 °C RMSE; post-calibration 2.341 °C MAE / 2.907 °C RMSE; all against a simplified modeled radiative emission proxy, not direct roof thermometer readings.
- **Uncertainty and representativeness:** native Landsat thermal support ~100 m; scene uncertainty about 3 K; simplified geometry/material/emissivity; time interpolation; no independent rooftop observations, no demonstrated cool-roof intervention outcome.
- **Reproducibility gap:** original ten-date archive includes summary tables only, not all weather inputs, per-date surfaces, QA matching and logs. Inputs can be fetched and runs reproduced online; compare hashes and dates before treating as reproduced.

**Suitable claim:** 'We compared simulated urban surface thermal proxies against independent Landsat satellite retrievals on ten acquisition dates and reported exploratory scene-level differences and their limitations.'

**Not suitable:** 'UrbanRay predicts real roof temperature to within 2.34 °C' or 'Reflective roofs are proven to cool real NYC roofs by the simulated amount.'

**Audit source:** `outputs/multidate_validation/audit/AUDIT_REPORT.md`. **Do not use synthetic fixtures in `tests/` as observed environmental data.**
