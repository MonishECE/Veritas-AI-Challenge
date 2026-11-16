# URBANRAY AI — Geometry-Aware Urban Reflection Screening

**Veritas AI 2026 | Climate Track | Offline research prototype**

**The problem.** Reflective urban roofs may reduce their own absorbed sunlight while also redirecting sunlight toward adjacent facades. Direct-sun models miss some of that geometry-dependent exposure, while full multibounce calculations can be difficult to interpret at building level.

**What we built.** We convert 88 documented NYC building footprints into 2,082 simplified modeled patches. A multibounce physics simulator generates shortwave radiation training targets. We train two distinct geometry-aware gradient-boosted specialist models: (1) absorbed shortwave prediction and (2) modeled wall-reflection hotspot ranking. One decision engine displays both outputs, building screening priorities, model disagreement, and required review flags.

**Simulator-referenced evaluation.** On a previously inspected 2025-06-23 date with withheld spatial blocks, regression MAE is **3.08 W/m²** versus **7.90 W/m²** for no reflection. The specialist captures **85.88%** of top-20 simulated wall reflection exposure versus **11.46%** for a direct-exposure rule. Our top-3/top-5/top-10 simulated roof-portion priority selections **did not** improve on the simpler baseline. All these benchmarks use approximate simulator targets, not observed wall irradiance.

**Exploratory observational comparison.** Five previously reported 2025 Landsat development-date summaries yield equally weighted grid-cell MAEs of **6.23°C** (AI), **6.17°C** (physics), and **6.32°C** (no reflection), on 135 correlated matched grid-cell rows. Original raw date-level outputs are **not in this curated archive**; the aggregate is not an independently reproduced result. This comparison uses a thermal emission proxy, **not** roof thermometers or measured roof cooling. A separate 2026 Landsat result is physics-only.

**How to see it.** Extract the ZIP and open `outputs/unified_decision_engine/dashboard/UrbanRay_AI_Explorer_OFFLINE.html`. Click *Reset to demo scene* and inspect the roof and wall decision panel and specialist map. All data and charts are embedded; no account or hosting needed.

**Limits.** The all-roofs counterfactual cannot identify one roof's causal effect on a wall. Model disagreement is not calibrated uncertainty; roof intervention outcomes, operational financial savings, field facade hotspot accuracy and cross-city transfer remain unproven. This is research screening, not installation approval.

**Source context.** NYC Open Data building geometry, Open-Meteo historical gridded weather, USGS Landsat products, scikit-learn, and Plotly. Full technical source at `UNIFIED_DECISION_ENGINE_REPORT.md` and `COMPETITION_EVIDENCE_AUDIT.md`.
