# Competition evidence audit — UrbanRay

This report independently checks the arithmetic and embedded contents of the **packaged** artifacts. It does **not** rerun the original simulator, reproduce raw Landsat pairings, or certify field accuracy.

## Verified packaged facts
- The unified offline dashboard contains **88 buildings**, **2,082** mesh patches, **24 hours** per scenario, and distinct AI, specialist, physics and no-reflection predictions.
- Both trained model file SHA-256 values match the saved unified decision report.
- On simulator-referenced top-20 modeled wall reflection exposure, specialist: **85.88%**, direct-exposure rule: **11.46%**. No wall-sensor validation.
- On simulator-referenced roof-portion selection, AI offers **no demonstrated improvement** over no-reflection at K = 3, 5, 10.
- Five previously reported 2025 development scenes cover **135 correlated 30 m output-grid rows**, not independent roof thermometers.
- Equal-scene mean **grid-cell MAE** against Landsat surface-temperature retrievals: decision_ai **6.227 °C**, physics **6.169 °C**, no_reflection **6.323 °C**.
- AI beats no-reflection in **3/5** scenes and physics in **0/5** scenes. No significant improvement claim.

## Evidence hierarchy and limits
1. **Software/data-integrity checks:** verified using files bundled in this ZIP; source tests reported separately.
2. **Simulation-relative AI validation:** performed against approximate physics labels on a previously inspected date and spatial subset. Do not call this independent field validation.
3. **Five-date AI/physics/Landsat proxy summary:** only reported aggregate JSONs are included, so raw weather, face-hour records and per-date matching logs cannot be independently replayed here.
4. **Separate exploratory 2026 Landsat evidence:** archived physics-only proxy versus retrieval; do not attribute it to the unified AI.
5. **No measured intervention:** roof cooling, energy-bill savings, field reflection hotspots, calibrated uncertainty, and single-roof causal effects remain unproven.

The comparison units differ: radiation model errors are W/m², reported Landsat differences are °C, and rooftop screening reports kWh of **solar heat**, not electrical savings.
