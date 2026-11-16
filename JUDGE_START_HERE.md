# UrbanRay AI | Judge start here

**Project:** UrbanRay AI — Unified Urban Radiation & Reflection Decision Engine  
**Competition:** Veritas AI 2026 · Climate track  
**Development period (team-reported):** November 14–15, 2026. The team reports that the complete project was created during the event. ZIP snapshots are iterations from that period, not separate submissions. Original development records and dates should be retained if requested by organizers.

## Start the demo (no installation)

1. Extract the ZIP and open `outputs/unified_decision_engine/dashboard/UrbanRay_AI_Explorer_OFFLINE.html`.
2. Select **Reset to demo scene**, then **Hotspot specialist · modeled reflections**.
3. Select a building from the map or ranked table. Inspect its **simulated** roof absorbed-solar-heat difference, surrounding-facade indicators, disagreement and review flag.
4. For a two-minute walkthrough, follow `JUDGE_DEMO_2MIN.md`.

## One-sentence contribution

UrbanRay integrates two trained geometry-aware AI specialists and an approximate physics simulator in one research-screening dashboard to explore how reflective-roof scenarios may change rooftop solar absorption and neighboring-building solar reflections.

## What is established

| Claim | Evidence | Important boundary |
|---|---|---|
| Geometry-aware radiation model: 3.081 W/m² MAE vs 7.898 W/m² no-reflection | Held-out **simulator** face-hour benchmark | Not measured roof-temperature accuracy |
| Hotspot specialist captures 85.88% of simulator top-20 reflection exposure vs 11.46% direct-exposure rule | Simulator-referenced hotspot selection | No independent wall sensor confirmation |
| AI rooftop portion selections at K=3/5/10 | Archived decision comparison | No demonstrated improvement over the simple baseline |
| 2025 five-date Landsat comparison: 6.227°C AI, 6.169°C physics, 6.323°C no-reflection | Reported, scene-equal mean grid-cell MAE across 135 correlated matched cells | Aggregate summaries only; per-date raw validation records are not packaged; not individual-roof validation |

The separate 2026 historical Landsat results describe **physics-only** thermal-proxy comparison; do not present them as validation of the unified AI. Dashboard interactions replay saved simulated results and **do not compute a new field experiment**.

## Reproducibility

From this folder with Python 3.10+ and dependencies installed (`python -m pip install -r requirements.txt`):

```bash
python -m pytest -q tests
python verify_checksums.py
python competition_evidence_audit.py
python decision_validation_gate.py
```

See `outputs/decision_validation_gate/report.md` for the complete decision-evidence result. These tests check code and packaged evidence; they do **not** establish real-world intervention outcomes. The roughly 131 MB face-hour training dataset and raw weather files are not included in this curated package, so full historical retraining is not reproducible offline from it alone.

## How to interpret the decision panel

- **Roof benefit** is modeled absorbed *solar heat* (kWh), not electricity or bill savings.
- **Nearby facade change** describes the counterfactual where **all roofs** change; it is *not caused by changing the selected roof alone*.
- **Review flags** are research heuristics, not calibrated risk probabilities or engineering approvals.
- The interface is a prototype for screening research priorities, **not an installation recommendation**.

- `DATA_SOURCES_AND_CREDITS.md`: official dataset links, source traceability, package inventory, and attribution.

## Other documents

- `README.md`: current application and setup
- `JUDGE_DEMO_2MIN.md`: speaking script
- `COMPETITION_EVIDENCE_AUDIT.md`: packaged evidence audit
- `SCIENTIFIC_STATUS.md`: research findings and limitations
- `FINAL_SUBMISSION_GUIDE.md`: reproduction and additional commands
- `AI_ASSISTANCE_CREDITS.md`: AI assistance and authorship disclosure

**Competition eligibility:** The team reports that the entire project was developed inside the hackathon window. Independent confirmation of every source file's development timestamp is not provided by this ZIP. Preserve truthful records and confirm requirements with the organizer rather than inferring eligibility from wording alone.
