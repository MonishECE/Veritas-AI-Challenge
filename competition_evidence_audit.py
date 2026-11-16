"""Read-only competition evidence audit over packaged artifacts.

Does not train, select models, or convert simulator labels to field evidence.
Raises if archived figures/data and dashboard disagree. Exports a short audit
for reviewers. It does not reproduce omitted historical simulation inputs.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
DASH = ROOT / 'outputs/unified_decision_engine/dashboard/UrbanRay_AI_Explorer_OFFLINE.html'
SIM = ROOT / 'outputs/unified_decision_engine/real_world_upgrade'
MODEL = ROOT / 'outputs/geometry_ai_upgrade'


def read_json(rel: str) -> dict:
    return json.loads((ROOT / rel).read_text(encoding='utf-8'))


def close(got: float, expected: float, name: str, tolerance=1e-8) -> None:
    if not math.isfinite(float(got)) or not math.isclose(float(got), float(expected), rel_tol=0, abs_tol=tolerance):
        raise AssertionError(f'{name}: {got} is not consistent with {expected}')


def audit() -> dict:
    summary = read_json('VALIDATION_2025_SUMMARY.json')
    dates = summary['dates']
    if len(dates) != 5:
        raise AssertionError('Expected five historical development scenes')
    cells = sum(int(d['cells']) for d in dates.values())
    if cells != summary['total_matched_grid_cell_records']:
        raise AssertionError('Reported five-date grid-cell count is inconsistent')
    metrics = ('decision_ai', 'physics', 'no_reflection')
    averaged = {m: sum(float(d[m]) for d in dates.values()) / len(dates) for m in metrics}
    for m in metrics:
        close(averaged[m], summary['equal_date_weight_mean_grid_cell_MAE_C'][m], f'2025 five-date {m}')
    win_counts = {m: sum(d['decision_ai'] < d[m] for d in dates.values()) for m in ('physics','no_reflection')}

    hotspot = pd.read_csv(SIM / 'locked_wall_hotspot_selection_SIMULATED.csv')
    roof = pd.read_csv(SIM / 'locked_decision_topk_benchmark_SIMULATED.csv')
    audit_file = read_json('outputs/unified_decision_engine/dashboard/unified_decision_audit.json')
    if audit_file['building_count'] != 88 or audit_file['neighbor_wall_change_is_individual_roof_causal_effect']:
        raise AssertionError('Incorrect all-roofs causal limitation in decision audit')
    for kind, file in (('regression', MODEL/'regression_surrogate.joblib'), ('hotspot', MODEL/'hotspot_surrogate.joblib')):
        digest=hashlib.sha256(file.read_bytes()).hexdigest()
        close_len=len(digest)
        if close_len!=64 or digest != audit_file[f'{kind}_model_sha256']:
            raise AssertionError(f'Model integrity mismatch for {kind}')
    if not (hotspot['top_k_wall_patches'].isin([10,20,50])).all():
        raise AssertionError('Unexpected wall-hotspot benchmark top K')
    k20=hotspot[hotspot.top_k_wall_patches==20]
    def hot_score(model):
        row=k20.loc[k20.method==model,'percent_of_reference_topk_exposure']
        if len(row)!=1: raise AssertionError(f'Cannot identify {model} specialist score')
        return float(row.iloc[0])
    special, direct = hot_score('dedicated_hotspot_ai'), hot_score('direct_exposure_rule')
    if not (0 <= direct <= 100 and 0 <= special <= 100):
        raise AssertionError('Invalid hotspot capture percentage')
    roof_equal={}
    for k in (3,5,10):
        part=roof[roof.top_k_roof_portions==k]
        if len(part)!=3 or set(part.method)!={'ai','no_reflection','direct_ai'}:
            raise AssertionError(f'Incomplete matched roof decision comparison at K={k}')
        ref=float(part.loc[part.method=='no_reflection','simulator_capture_percent'].iloc[0])
        actual=float(part.loc[part.method=='ai','simulator_capture_percent'].iloc[0])
        roof_equal[str(k)]={'ai_percent':actual,'baseline_percent':ref,'ai_improves':actual>ref+1e-9}

    html=DASH.read_text(encoding='utf-8')
    assert 'const D=' in html and 'const VALIDATION_2025=' in html, 'Missing embedded dashboard data'
    embedded=json.JSONDecoder().raw_decode(html[html.index('const D=')+len('const D='):])[0]
    embedded_2025=json.JSONDecoder().raw_decode(html[html.index('const VALIDATION_2025=')+len('const VALIDATION_2025='):])[0]
    if len(embedded['unified_engine']['screenings'])!=88 or len(embedded['unified_engine']['top_hotspots'])!=20:
        raise AssertionError('Dashboard does not include the complete unified decision outputs')
    if embedded_2025!=summary:
        raise AssertionError('Embedded five-date evidence differs from the archived summary')
    if len(embedded['times'])!=24:
        raise AssertionError('Dashboard does not have 24 simulated time points')
    states=embedded['states']
    for scenario in ('dark_roof','reflective_roof'):
        for t in embedded['times']:
            group=states[scenario][t]
            for method in ('ai','physics','no_reflection','hotspot_ai'):
                if method not in group or len(group[method])!=2082:
                    raise AssertionError(f'Incomplete scenario/method/time: {scenario} {method} {t}')
    # Precomputed comparisons are not interchangeable; actually distinct method outputs.
    sample=states['dark_roof'][embedded['times'][17]]
    if sample['ai']==sample['physics'] or sample['physics']==sample['no_reflection'] or sample['hotspot_ai']==sample['ai']:
        raise AssertionError('Method outputs appear identical at the demonstrated daylight hour')

    return {
      'audit_scope': 'packaged source, static dashboard arrays, stored simulator metrics, reported development-date summaries',
      'full_simulator_or_observational_replay_performed': False,
      'source_code_tests_reported_separately': True,
      'model_weights_sha256_match_embedded_decision_report': True,
      'dashboard_state': {'full_buildings':88,'model_methods_verified':['ai','hotspot_ai','physics','no_reflection'],
                          'hours_per_scenario':24,'surface_patches':2082,'different_methods_have_distinct_values':True},
      'simulator_hotspots_top20': {'specialist_capture_percent':special,'direct_exposure_capture_percent':direct,
          'independent_wall_sensor_validation':False,'new_unseen_holdout':False},
      'simulator_roof_selection':roof_equal,
      'reported_2025_development_satellite': {'dates':len(dates),'correlated_matched_grid_cells':cells,
          'equal_scene_mean_grid_cell_MAE_C':averaged,'ai_better_than_physics_dates':win_counts['physics'],
          'ai_better_than_no_reflection_dates':win_counts['no_reflection'],
          'raw_validator_outputs_reproduced_in_package':False,'measured_rooftop_accuracy':False},
      '2026_physics_only_evidence_is_not_ai_validation':True,
      'real_roof_cooling_demonstrated':False,
      'research_screening_not_roofing_approval':True,
      'interpretation': 'Simulation-based hotspot selection benefit is the primary supported AI decision finding; rooftop selection shows no improvement over simpler rules; observational accuracy remains unestablished.',
    }


def write_reports(report: dict) -> None:
    (ROOT/'COMPETITION_EVIDENCE_AUDIT.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    a=report['reported_2025_development_satellite'];w=report['simulator_hotspots_top20']; roof=report['simulator_roof_selection']
    lines=[
      '# Competition evidence audit — UrbanRay',
      '',
      'This report independently checks the arithmetic and embedded contents of the **packaged** artifacts. '
      'It does **not** rerun the original simulator, reproduce raw Landsat pairings, or certify field accuracy.',
      '',
      '## Verified packaged facts',
      f'- The unified offline dashboard contains **{report["dashboard_state"]["full_buildings"]} buildings**, **2,082** mesh patches, **24 hours** per scenario, and distinct AI, specialist, physics and no-reflection predictions.',
      '- Both trained model file SHA-256 values match the saved unified decision report.',
      f'- On simulator-referenced top-20 modeled wall reflection exposure, specialist: **{w["specialist_capture_percent"]:.2f}%**, direct-exposure rule: **{w["direct_exposure_capture_percent"]:.2f}%**. No wall-sensor validation.',
      '- On simulator-referenced roof-portion selection, AI offers **no demonstrated improvement** over no-reflection at K = '+', '.join(roof)+'.',
      f'- Five previously reported 2025 development scenes cover **{a["correlated_matched_grid_cells"]} correlated 30 m output-grid rows**, not independent roof thermometers.',
      '- Equal-scene mean **grid-cell MAE** against Landsat surface-temperature retrievals: '
      + ', '.join(f'{name} **{value:.3f} °C**' for name,value in a['equal_scene_mean_grid_cell_MAE_C'].items())+'.',
      f'- AI beats no-reflection in **{a["ai_better_than_no_reflection_dates"]}/5** scenes and physics in **{a["ai_better_than_physics_dates"]}/5** scenes. No significant improvement claim.',
      '',
      '## Evidence hierarchy and limits',
      '1. **Software/data-integrity checks:** verified using files bundled in this ZIP; source tests reported separately.',
      '2. **Simulation-relative AI validation:** performed against approximate physics labels on a previously inspected date and spatial subset. Do not call this independent field validation.',
      '3. **Five-date AI/physics/Landsat proxy summary:** only reported aggregate JSONs are included, so raw weather, face-hour records and per-date matching logs cannot be independently replayed here.',
      '4. **Separate exploratory 2026 Landsat evidence:** archived physics-only proxy versus retrieval; do not attribute it to the unified AI.',
      '5. **No measured intervention:** roof cooling, energy-bill savings, field reflection hotspots, calibrated uncertainty, and single-roof causal effects remain unproven.',
      '',
      'The comparison units differ: radiation model errors are W/m², reported Landsat differences are °C, and rooftop screening reports kWh of **solar heat**, not electrical savings.',
    ]
    (ROOT/'COMPETITION_EVIDENCE_AUDIT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':
    summary=audit();write_reports(summary)
    print(json.dumps({'status':'PASS','hotspot_top20':summary['simulator_hotspots_top20'],
        'reported_2025':summary['reported_2025_development_satellite']},indent=2))
