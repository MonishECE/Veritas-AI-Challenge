"""Audit a *decision*, not only prediction MAE, on LOCKED simulator-held-out roofs.

The units are kWh of modeled absorbed SOLAR HEAT avoided over one 24-hour
simulation, not electricity saved, greenhouse-gas reductions, or observed
cooling. The spatial-block split holds only some roof faces per building;
therefore each 'building' below is a spatially held-out roof *portion*.
The reference for all decision regret is the simulator, not field truth.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))
from integrated88 import load_verified_geometry
from urbanray import build_scene

SCENARIOS = ('dark_roof', 'reflective_roof')
PREDICTIONS = {
    'physics': 'physics_label_Wm2',
    'ai': 'pred_Reflection-residual_AI',
    'no_reflection': 'pred_No-reflection_physics',
    'direct_ai': 'pred_Direct_AI',
}


def build_decision_audit(df: pd.DataFrame, area_m2: np.ndarray,
                         k_values=(3, 5, 10)) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    required = {'building_id', 'kind', 'face_id', 'time_utc', 'scenario', *PREDICTIONS.values()}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f'Missing held-out data columns: {sorted(missing)}')
    faces = df.loc[(df.kind == 'roof') & (df.building_id.astype(str) != 'GROUND')].copy()
    if faces.empty:
        raise ValueError('No held-out roofs to audit')
    faces['building_id'] = faces.building_id.astype(str)
    faces.face_id = pd.to_numeric(faces.face_id, errors='raise').astype(int)
    if faces.face_id.min() < 0 or faces.face_id.max() >= len(area_m2):
        raise ValueError('Face index outside the source geometry')
    if faces.duplicated(['building_id','face_id','time_utc','scenario']).any():
        raise ValueError('Duplicate face/time/scenario in locked holdout')
    if set(faces.scenario.unique()) != set(SCENARIOS):
        raise ValueError('Both roof-albedo simulations are required')
    for col in PREDICTIONS.values():
        if not np.isfinite(faces[col]).all():
            raise ValueError(f'Nonfinite prediction in {col}')
    joined = faces.pivot(index=['building_id','face_id','time_utc'],
                         columns='scenario', values=list(PREDICTIONS.values()))
    if joined.isna().any().any():
        raise ValueError('Unpaired or incomplete roof scenarios')
    ids = joined.index.get_level_values('face_id').to_numpy(dtype=int)
    area = np.asarray(area_m2, dtype=float)[ids]
    if not np.isfinite(area).all() or np.any(area <= 0):
        raise ValueError('Invalid geometry face areas')
    # One 1-hour simulated sample contributes (W/m²) * m² * 1 h / 1000 Wh/kWh.
    savings = pd.DataFrame(index=joined.index)
    for method, col in PREDICTIONS.items():
        savings[f'{method}_solar_heat_avoided_kWh_day'] = (
            joined[(col,'dark_roof')].to_numpy() -
            joined[(col,'reflective_roof')].to_numpy()
        ) * area / 1000.
    savings = savings.groupby(level='building_id', sort=True).sum().reset_index()
    coverage = faces[faces.scenario == 'dark_roof'].groupby('building_id').agg(
        heldout_roof_faces=('face_id','nunique'), observed_simulated_hours=('time_utc','nunique'))
    patch_area = faces[faces.scenario == 'dark_roof'].drop_duplicates(['building_id','face_id'])
    coverage['heldout_roof_portion_m2'] = patch_area.groupby('building_id').apply(
        lambda z: float(np.sum(np.asarray(area_m2)[z.face_id.to_numpy(dtype=int)])),
        include_groups=False)
    savings = savings.join(coverage, on='building_id')
    if (savings.observed_simulated_hours != 24).any():
        raise ValueError('This decision benchmark requires exactly 24 simulated hourly samples')
    reference = savings.set_index('building_id')['physics_solar_heat_avoided_kWh_day']
    records = []
    for k in sorted(set(int(v) for v in k_values)):
        if k <= 0 or k > len(savings):
            continue
        oracle = savings.nlargest(k, 'physics_solar_heat_avoided_kWh_day')
        oracle_total = float(oracle.physics_solar_heat_avoided_kWh_day.sum())
        for method in ('ai','no_reflection','direct_ai'):
            selected = savings.sort_values([f'{method}_solar_heat_avoided_kWh_day','building_id'],
                ascending=[False,True]).head(k)
            true_total = float(selected.physics_solar_heat_avoided_kWh_day.sum())
            oracle_ids = set(oracle.building_id)
            selected_ids = set(selected.building_id)
            records.append({'method':method,'top_k_roof_portions':k,
                'candidate_roof_portions':int(len(savings)),
                'selection_unit':'spatially held-out ROOF PORTIONS; not complete independent buildings',
                'simulated_reference_avoided_solar_heat_kWh_day':true_total,
                'simulator_oracle_kWh_day':oracle_total,
                'simulator_regret_kWh_day':max(0.,oracle_total-true_total),
                'simulator_capture_percent':100*true_total/oracle_total if oracle_total > 0 else None,
                'top_k_overlap_with_simulator_oracle':len(selected_ids & oracle_ids),
                'selected_building_ids':'|'.join(selected.building_id)})
    decision = pd.DataFrame(records)
    meta = {'scope':'Locked date + spatial-block held-out simulation labels only',
        'physical_target':'modeled shortwave radiation absorbed by roof surfaces',
        'impact_metric':'kWh of modeled absorbed SOLAR HEAT avoided on the modeled date, NOT kWh electricity or utility bill savings',
        'candidate_unit':'held-out roof portions of buildings (spatial blocks may cut roofs)',
        'building_portions':int(len(savings)),
        'simulated_hours':24,
        'reference':'approximate multibounce physics simulation, not measured roof cooling',
        'independent_observational_validation':False,
        'decision_advantage_is_not_guaranteed':True,
        'all_methods_evaluated_against_identical_locked_simulator_labels':True,
        'baseline_comparison':'same candidate roof portions and same top-k selection slots'}
    return savings, decision, meta


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--predictions',type=Path,default=ROOT/'outputs/ai_upgrade/reflection_residual/candidate_heldout_predictions.csv')
    p.add_argument('--out',type=Path,default=ROOT/'outputs/real_world_upgrade')
    a=p.parse_args()
    records, source_hash = load_verified_geometry(ROOT/'data/nyc_88_source.geojson')
    scene=build_scene(records,ground_cell_m=25,max_buildings=88)
    df=pd.read_csv(a.predictions,dtype={'building_id':str})
    savings,decision,meta=build_decision_audit(df,scene['area'])
    a.out.mkdir(parents=True,exist_ok=True)
    savings.to_csv(a.out/'heldout_roof_portion_solar_heat_savings_SIMULATED.csv',index=False)
    decision.to_csv(a.out/'locked_decision_topk_benchmark_SIMULATED.csv',index=False)
    meta.update(geometry_sha256=source_hash,
                heldout_predictions_sha256=hashlib.sha256(a.predictions.read_bytes()).hexdigest())
    (a.out/'decision_audit.json').write_text(json.dumps(meta,indent=2))
    print(decision[['method','top_k_roof_portions','simulator_capture_percent','simulator_regret_kWh_day']].to_string(index=False))
    return decision

if __name__=='__main__':main()
