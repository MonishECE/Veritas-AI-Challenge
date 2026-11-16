"""Reproducible, conservative audit of *decisions* using archived UrbanRay evidence.

This does not turn simulation outputs or Landsat-temperature proxies into real
intervention observations. It intentionally never loads untrusted joblib files.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
ROOF = ROOT / 'outputs/real_world_upgrade/heldout_roof_portion_solar_heat_savings_SIMULATED.csv'
WALL = ROOT / 'outputs/real_world_upgrade/locked_wall_hotspot_selection_SIMULATED.csv'
SAT = ROOT / 'VALIDATION_2025_SUMMARY.json'
METHOD_COLUMNS = {
    'ai':'ai_solar_heat_avoided_kWh_day',
    'no_reflection':'no_reflection_solar_heat_avoided_kWh_day',
    'direct_ai':'direct_ai_solar_heat_avoided_kWh_day',
}
ORACLE = 'physics_solar_heat_avoided_kWh_day'


def sha256(path:Path)->str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rank(df:pd.DataFrame, column:str, k:int)->list[str]:
    return (df.sort_values([column,'building_id'],ascending=[False,True],kind='stable')
            .head(k)['building_id'].tolist())


def audit_roofs(df:pd.DataFrame, ks=(3,5,10))->list[dict]:
    required={'building_id',ORACLE,*METHOD_COLUMNS.values(),'observed_simulated_hours','heldout_roof_portion_m2'}
    if required-set(df.columns):
        raise ValueError(f'Missing roof columns: {sorted(required-set(df.columns))}')
    df=df.copy()
    df['building_id']=df['building_id'].astype(str)
    if df['building_id'].duplicated().any() or df.empty:
        raise ValueError('Roof building IDs must be nonempty and unique')
    if not np.isfinite(df[[ORACLE,*METHOD_COLUMNS.values()]].to_numpy(dtype=float)).all():
        raise ValueError('Missing/invalid roof predictions')
    if (df.observed_simulated_hours!=24).any() or (df.heldout_roof_portion_m2<=0).any():
        raise ValueError('Incomplete simulated day or invalid area')
    truth=dict(zip(df.building_id,df[ORACLE]))
    out=[]
    for k in ks:
        if not 1<=k<=len(df):
            raise ValueError(f'Invalid k={k}')
        oracle=rank(df,ORACLE,k)
        oracle_value=sum(truth[i] for i in oracle)
        selected={name:rank(df,col,k) for name,col in METHOD_COLUMNS.items()}
        base=selected['no_reflection']; base_value=sum(truth[i] for i in base)
        for name,ids in selected.items():
            value=sum(truth[i] for i in ids)
            out.append({
                'k':k,'method':name,'candidate_roof_portions':len(df),
                'selected_building_ids':ids,
                'selection_changed_vs_no_reflection':set(ids)!=set(base),
                'selected_set_overlap_vs_no_reflection':len(set(ids)&set(base)),
                'oracle_set_overlap':len(set(ids)&set(oracle)),
                'simulator_oracle_kWh_solar_heat':oracle_value,
                'simulator_selected_kWh_solar_heat':value,
                'simulator_regret_kWh_solar_heat':max(0.0,oracle_value-value),
                'gain_vs_no_reflection_kWh_solar_heat':value-base_value,
                'outperforms_no_reflection_on_simulated_decision':value>base_value+1e-9,
                'observed_intervention_benefit':False,
            })
    return out


def audit_wall(df:pd.DataFrame)->list[dict]:
    required={'method','top_k_wall_patches','percent_of_reference_topk_exposure',
              'num_spatially_heldout_wall_patches','selected_face_ids'}
    if required-set(df.columns): raise ValueError('Missing wall selection fields')
    out=[]
    for k, group in df.groupby('top_k_wall_patches'):
        lookup=group.set_index('method')
        if not {'ai_residual','direct_exposure_rule'}.issubset(lookup.index):
            raise ValueError(f'Missing comparator at k={k}')
        specialist=lookup.loc['ai_residual'];baseline=lookup.loc['direct_exposure_rule']
        ids_a=set(str(specialist.selected_face_ids).split('|'))
        ids_b=set(str(baseline.selected_face_ids).split('|'))
        if len(ids_a)!=int(k) or len(ids_b)!=int(k):
            raise ValueError('Duplicate/invalid wall patch selection')
        out.append({'k':int(k),'candidate_wall_patches':int(specialist.num_spatially_heldout_wall_patches),
                    'specialist_capture_percent_simulated':float(specialist.percent_of_reference_topk_exposure),
                    'direct_exposure_capture_percent_simulated':float(baseline.percent_of_reference_topk_exposure),
                    'specialist_gain_percentage_points_simulated':float(specialist.percent_of_reference_topk_exposure-baseline.percent_of_reference_topk_exposure),
                    'selected_set_overlap':len(ids_a&ids_b),
                    'selection_changed':ids_a!=ids_b,
                    'independently_observed_wall_reflection':False})
    return sorted(out,key=lambda r:r['k'])


def audit_satellite(data:dict)->dict:
    if 'dates' not in data or len(data['dates'])<2:raise ValueError('At least two scenes required')
    scenes=[]
    for date,record in sorted(data['dates'].items()):
        required={'decision_ai','physics','no_reflection','cells'}
        if not required.issubset(record):raise ValueError(f'Missing scene data {date}')
        if record['cells']<=0 or not np.isfinite([record[k] for k in ('decision_ai','physics','no_reflection')]).all():
            raise ValueError(f'Invalid scene data {date}')
        scenes.append({'date':date,'cells':int(record['cells']),
            'ai_mae_C':float(record['decision_ai']),
            'physics_mae_C':float(record['physics']),
            'no_reflection_mae_C':float(record['no_reflection']),
            'ai_minus_no_reflection_mae_C':float(record['decision_ai']-record['no_reflection']),
            'ai_minus_physics_mae_C':float(record['decision_ai']-record['physics'])})
    out={'source_status':data.get('status','UNKNOWN'), 'source':data.get('source','UNKNOWN'),
         'n_development_scenes':len(scenes),'correlated_grid_cell_rows':sum(s['cells'] for s in scenes),
         'scene_records':scenes,'means_C':{},'ai_better_than_no_reflection_scenes':sum(s['ai_minus_no_reflection_mae_C']<0 for s in scenes),
         'ai_better_than_physics_scenes':sum(s['ai_minus_physics_mae_C']<0 for s in scenes),
         'independent_new_scene_test':False,'individual_roof_accuracy_established':False,
         'observed_decision_success_established':False}
    for k in ('ai_mae_C','physics_mae_C','no_reflection_mae_C','ai_minus_no_reflection_mae_C','ai_minus_physics_mae_C'):
        out['means_C'][k]=float(np.mean([s[k] for s in scenes]))
    return out


def generate(roof=ROOF,wall=WALL,satellite=SAT)->dict:
    for p in (roof,wall,satellite):
        if not p.is_file():raise FileNotFoundError(p)
    roof_rows=audit_roofs(pd.read_csv(roof,dtype={'building_id':str}))
    wall_rows=audit_wall(pd.read_csv(wall))
    sat=audit_satellite(json.loads(satellite.read_text(encoding='utf8')))
    return {
        'report_version':'1.0',
        'inputs_sha256':{p.name:sha256(p) for p in (roof,wall,satellite)},
        'roof_selection_simulation':roof_rows,
        'wall_selection_simulation':wall_rows,
        'satellite_temperature_proxy_development':sat,
        'evidence_gate':{
            'simulator_wall_screening_improvement_shown':any(r['specialist_gain_percentage_points_simulated']>0 for r in wall_rows),
            'simulator_roof_screening_improvement_shown':any(r['method']=='ai' and r['outperforms_no_reflection_on_simulated_decision'] for r in roof_rows),
            'real_world_decision_improvement_shown':False,
            'reason':'No independently observed post-intervention outcomes or wall reflection ground truth; satellite scenes provide aggregate temperature proxy errors, not observed screening decisions.',
            'scope':'Retrospective, already-inspected simulator data and five development satellite scenes; not independent prospective validation.',
            'statistical_claim':'No independent scene-level uncertainty or significance claim; correlated spatial/temporal rows must not be treated as independent observations.'
        }
    }


def report_md(result:dict)->str:
    roof=result['roof_selection_simulation'];wall=result['wall_selection_simulation'];sat=result['satellite_temperature_proxy_development'];gate=result['evidence_gate']
    lines=['# UrbanRay decision evidence gate — reproducible audit','',
        '**Status: observational decision advantage NOT established.** This report recomputes decisions from packaged evidence, not original physical experiments.','',
        '## Rooftop selection (simulator-relative, held-out roof portions)','',
        '| K | AI and no-reflection choose same set? | AI gain over no-reflection (modeled solar heat kWh/day) |','|---:|:---:|---:|']
    for r in roof:
        if r['method']=='ai':lines.append(f"| {r['k']} | {'No' if r['selection_changed_vs_no_reflection'] else 'Yes'} | {r['gain_vs_no_reflection_kWh_solar_heat']:.4f} |")
    lines+=['','These are **roof portions** and an all-roofs-change scenario; no individual building intervention impact is established.','',
        '## Reflection-hotspot selection (simulator-relative wall patches)','',
        '| K | Specialist captured | Direct-exposure rule captured | Difference (percentage points) |', '|---:|---:|---:|---:|']
    for r in wall:lines.append(f"| {r['k']} | {r['specialist_capture_percent_simulated']:.2f}% | {r['direct_exposure_capture_percent_simulated']:.2f}% | {r['specialist_gain_percentage_points_simulated']:.2f} |")
    lines+=['','This is a comparison against simulator labels, not independent wall-sensor observations.','',
        '## Satellite temperature-proxy development scenes','',
        f"There are **{sat['n_development_scenes']}** already-inspected scenes ({sat['correlated_grid_cell_rows']} correlated grid cells). Scene-equal mean grid-cell MAEs: AI **{sat['means_C']['ai_mae_C']:.3f}°C**, physics **{sat['means_C']['physics_mae_C']:.3f}°C**, no-reflection **{sat['means_C']['no_reflection_mae_C']:.3f}°C**.",
        f"AI is lower-error on **{sat['ai_better_than_no_reflection_scenes']}/{sat['n_development_scenes']}** scenes versus no-reflection and **{sat['ai_better_than_physics_scenes']}/{sat['n_development_scenes']}** versus physics.",
        '', '## Evidence gate','',
        f"- Simulator wall-hotspot advantage: **{'yes' if gate['simulator_wall_screening_improvement_shown'] else 'no'}**.",
        f"- Simulator roof selection advantage at reported K: **{'yes' if gate['simulator_roof_screening_improvement_shown'] else 'no'}**.",
        '- Real-world screening improvement: **NOT SHOWN**.',
        f"- Why: {gate['reason']}",
        '', '## How to reproduce','',
        'Run `python decision_validation_gate.py` from the package root. The script reads three included evidence files, checks required fields and integrity, and rewrites `outputs/decision_validation_gate/report.json` and `report.md`. It does not execute models, download data, or reinterpret simulator labels as observations.',
        '', '## What would actually close the gap','',
        '1. Before collecting new evidence, register a fixed decision target and evaluation unit, independent baseline, and the K values.',
        '2. Freeze model weights and screening rules. Reserve **new** dates and geographical blocks not previously used for model selection.',
        '3. Obtain independent observed **wall shortwave irradiance/reflection** for hotspot screening; for cool-roof decisions, obtain comparable measured before/after roof and ambient observations with appropriate controls.',
        '4. Evaluate *paired decisions at the same budget* against observed outcomes, with scene/building-level clustering and uncertainty; do not use correlated hourly patches as independent samples.',
        '5. Keep single-roof causal effects separate from all-roofs simulations, and disclose any failure to beat baselines.',
        '', 'The included Landsat proxy scenes cannot retroactively establish actual cooling or observed screening success.','']
    return '\n'.join(lines)


def main()->None:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',type=Path,default=ROOT/'outputs/decision_validation_gate')
    args=ap.parse_args()
    result=generate()
    args.out.mkdir(parents=True,exist_ok=True)
    (args.out/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf8')
    (args.out/'report.md').write_text(report_md(result),encoding='utf8')
    print(json.dumps(result['evidence_gate'],indent=2))

if __name__=='__main__':main()
