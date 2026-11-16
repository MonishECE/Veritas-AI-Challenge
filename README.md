# UrbanRay AI — Urban Radiation and Reflection Decision Engine

**Veritas AI 2026 Climate Track | research prototype | offline demo**

**For judges:** See [`JUDGE_START_HERE.md`](JUDGE_START_HERE.md) for the demo and evidence overview, [`DATA_SOURCES_AND_CREDITS.md`](DATA_SOURCES_AND_CREDITS.md) for datasets and websites, and [`AI_ASSISTANCE_CREDITS.md`](AI_ASSISTANCE_CREDITS.md) for ChatGPT contributions.

## Open the submission

1. Extract this entire ZIP to a normal folder.
2. Double-click **`outputs/unified_decision_engine/dashboard/UrbanRay_AI_Explorer_OFFLINE.html`**. This is the full interactive offline app: no Python, keys or hosted services required.
3. Click **Reset to demo scene**, select a building or use the ranking table, then choose **Hotspot specialist · modeled reflections**.
4. See [`JUDGE_DEMO_2MIN.md`](JUDGE_DEMO_2MIN.md) for a concise demo and [`SCIENTIFIC_STATUS.md`](SCIENTIFIC_STATUS.md) for limitations.

## Scientific contribution

We combine two separately trained geometry-aware gradient-boosting specialists behind one shared feature and decision API. The radiation regression specialist estimates simulated absorbed shortwave flux. The hotspot specialist ranks modeled facade reflection. Our decision layer displays roof solar-heat screening benefits, nearby-facade changes in an **all-roofs-changed counterfactual**, model disagreement, and review flags. The app does **not** approve installations.

### What our existing evidence supports

- **Against held-out simulator labels, not observations:** regression MAE **3.08 W/m²**, versus **7.90 W/m²** for a no-reflection baseline (one previously inspected June 23, 2025 date with withheld spatial blocks).
- **Simulator-based wall-reflection screening:** the dedicated hotspot specialist captures **85.9%** of modeled top-20 exposure on the historical held-out comparison; this is **not** independent wall-sensor validation.
- **Roof-portion priority selection:** the top-3/top-5/top-10 experiment did **not** improve on the simple baseline. We retain this negative result.
- **Five reported 2025 paired AI/physics/no-reflection Landsat development-date summaries:** equal-date mean grid-cell MAE **6.23 / 6.17 / 6.32 °C**, respectively, for **135 correlated matched grid-cell rows**. These reported aggregates were supplied separately; the per-date raw validation outputs are **not included** in this release, so they are not independently reproduced here. They do **not** establish field roof-temperature prediction or measured cooling.
- **Separate 2026 exploratory Landsat evidence:** archived physics-only surface-temperature-proxy comparison, not a verified AI evaluation.

## Reproduce checks from a fresh extraction

Python 3.10+ recommended; install dependencies in an isolated environment. For Windows PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pytest -q tests
.\.venv\Scripts\python.exe verify_checksums.py
.\.venv\Scripts\python.exe competition_evidence_audit.py
```

The **64 current automated tests** passed during this package verification. These test software and packaged scientific integrity, not new field accuracy. Training and full replay of the historical model cannot be completed offline from this curated ZIP because the original 131 MB face-hour training archive and raw historical weather are not bundled. Existing trained model assets and generated simulator benchmarks are included. See [`FINAL_SUBMISSION_GUIDE.md`](FINAL_SUBMISSION_GUIDE.md) for the smoke test and additional research instructions.

## Limitations and provenance

Our model outputs are simulated research-screening estimates. Neither a single-roof causal impact, observed roof cooling, calibrated confidence interval, production forecasting speedup, nor independently observed facade-hotspot accuracy has been demonstrated. The live dashboard's scenario values are not a new test set. All model-training targets and held-out evaluations are documented as simulated or exploratory, not measured truths. The project development dates are November 14–15, 2026, as reported by the team; repository history and development records provide the provenance.

## Where to look next

- [`JUDGE_DEMO_2MIN.md`](JUDGE_DEMO_2MIN.md): presentation script with key caveats.
- [`COMPETITION_EVIDENCE_AUDIT.md`](COMPETITION_EVIDENCE_AUDIT.md): machine-checkable results and evidence hierarchy.
- [`UNIFIED_DECISION_ENGINE_REPORT.md`](UNIFIED_DECISION_ENGINE_REPORT.md): architecture and simulator benchmarks.
- [`VALIDATION_2025_SUMMARY.json`](VALIDATION_2025_SUMMARY.json): explicitly transcribed five-date aggregate observations.
- [`FINAL_SUBMISSION_GUIDE.md`](FINAL_SUBMISSION_GUIDE.md): technical workflows; some historical simulation assets are not bundled in this curated ZIP.

External data and software-library attributions remain in the technical reports. Other required tool-use disclosures are provided in the separate competition credits section.

## New: reproducible decision-evidence gate (retrospective)

Run `python decision_validation_gate.py` to reproduce the packaged **decision comparisons** without retraining or modifying any models. Read `outputs/decision_validation_gate/report.md` and `report.json` for the complete method-level results, exact input hashes, and the evidence gate. The nine new `tests/test_decision_validation_gate.py` tests check decisions, comparator logic, and conservative interpretation. These comparisons use already-inspected simulator labels and five archived development satellite summaries. **They are not an independent real-world intervention validation.** The roof selection results do not beat the simpler baseline at K=3,5,10; simulated wall-hotspot screening improves substantially but has no independent wall-sensor check.
