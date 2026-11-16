# UrbanRay — two-minute Veritas judge demo

**0:00–0:20 · The problem**  In a dense city, reflective surfaces reduce their own solar absorption but also redirect sunlight toward other facades. A roof-only or direct-sun model can miss neighborhood effects.

**0:20–0:40 · Our system**  We use verified NYC footprints to build simplified geometry, run an approximate multibounce radiation model, and train two geometry-aware gradient-boosting specialists: absorbed shortwave flux and reflection hotspots. One decision interface shows both roles and flags disagreements. They are two separately trained models, not a merged set of weights.

**0:40–1:10 · Demonstration**  Open the single offline HTML. Click **Reset to demo scene** to show specialist reflection. Select a building and inspect its simulated roof-solar-heat reduction and nearby-wall change. Explain clearly that the wall number comes from **all roofs** changing together and cannot be attributed to that one roof.

**1:10–1:35 · Evidence**  On a previously inspected held-out simulator comparison, geometry-aware regression has 3.08 W/m² MAE vs 7.90 W/m² for a no-reflection baseline; the specialist's top-20 wall-exposure capture is 85.9%. The roof prioritization experiment did not beat the baseline. A separately reported five-date 2025 Landsat proxy comparison gives mean grid-cell errors of 6.23°C (AI), 6.17°C (physics), and 6.32°C (no reflection) and does not validate real rooftop temperatures.

**1:35–2:00 · Impact and limitations**  This is an explainable urban-heat *research screening* tool, not an engineering approval, an electricity-savings prediction, or measured retrofit impact. Next work requires independent physical observations, time-matched scene experiments, and single-roof interventions.

## High-probability judge questions

- **Why AI instead of simple physics?** It approximates geometry-dependent simulator outputs and helps rank modeled reflection hotspots; the tested AI is not demonstrated faster than the optimized physics solver.
- **Is there proof of real roof cooling?** No; modeled absorbed solar heat is not a field measurement of temperature or actual cooling.
- **How was leakage avoided?** The reported simulator experiment partitions by date and spatial blocks, but the test date has already been inspected; it must not be called a fresh unseen lockbox.
- **Are the models combined?** Yes, operationally in a shared decision interface; no, their parameters remain separately trained.
- **Does your AI always improve the decision?** No; rooftop top-K selection tied simpler baselines on the reported experiment. The evidence supports simulator-relative hotspot screening more strongly.
- **Can you reproduce the reported 2025 satellite aggregates?** The five date summaries can be recomputed from `VALIDATION_2025_SUMMARY.json`, but raw per-date matched outputs are not present in this curated ZIP. They must be provided for full observational reproduction.
