"""Test whether the residual surrogate identifies *simulated* reflected-sun hotspots.

New practical question: which individual facade patches receive the most extra
absorbed solar radiation due to urban reflections? Compare wall-face rankings
on the locked DATE + SPATIAL BLOCK holdout, not on the dashboard's whole day.
Reference is still the physics solver; no on-site measurements are implied.
"""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent

def score_hotspots(df: pd.DataFrame, k_values=(10,20,50)) -> tuple[pd.DataFrame,pd.DataFrame,dict]:
    required={'kind','face_id','time_utc','scenario','physics_label_Wm2',
        'pred_No-reflection_physics','pred_Reflection-residual_AI'}
    if not required.issubset(df.columns):
        raise ValueError(f'Missing columns: {required-set(df.columns)}')
    # Fix one albedo scenario to prevent multiple correlated copies of each face.
    d=df[(df['kind']=='wall') & (df['scenario']=='dark_roof')].copy()
    if d.empty or d.duplicated(['face_id','time_utc']).any():
        raise ValueError('Missing or duplicated locked wall face-hours')
    d['extra_reflected_physics_Wm2']=d.physics_label_Wm2-d['pred_No-reflection_physics']
    d['extra_reflected_ai_Wm2']=d['pred_Reflection-residual_AI']-d['pred_No-reflection_physics']
    additional={}
    for col,name in [('pred_Hotspot_AI','dedicated_hotspot_ai'),('pred_Legacy_residual_AI','previous_residual_ai')]:
        if col in d:
            metric='extra_'+name+'_Wm2'
            d[metric]=d[col]-d['pred_No-reflection_physics']
            additional[name]=metric
    # A simple rule using only direct+sky absorption; it does not estimate reflection.
    grouped=d.groupby('face_id').agg(
        truth_Wh_per_m2_day=('extra_reflected_physics_Wm2','sum'),
        ai_Wh_per_m2_day=('extra_reflected_ai_Wm2','sum'),
        direct_Wh_per_m2_day=('pred_No-reflection_physics','sum'),
        hours=('time_utc','nunique'),
    ).reset_index()
    for name,metric in additional.items():
        grouped[name+'_Wh_per_m2_day']=d.groupby('face_id')[metric].sum().reindex(grouped.face_id).to_numpy()
    if (grouped.hours != 24).any():
        raise ValueError('Need 24 hourly samples per held-out wall patch')
    cols=['truth_Wh_per_m2_day','ai_Wh_per_m2_day','direct_Wh_per_m2_day']
    if not np.isfinite(grouped[cols].to_numpy(dtype=float)).all():
        raise ValueError('Nonfinite wall reflection predictions')
    records=[]
    for k in sorted(set(k_values)):
        if k<1 or k>len(grouped):continue
        ideal=grouped.sort_values(['truth_Wh_per_m2_day','face_id'],ascending=[False,True]).head(k)
        upper=float(ideal.truth_Wh_per_m2_day.sum())
        options=[('ai_residual','ai_Wh_per_m2_day'),('direct_exposure_rule','direct_Wh_per_m2_day')]
        options += [(name,name+'_Wh_per_m2_day') for name in additional]
        for method,col in options:
            selected=grouped.sort_values([col,'face_id'],ascending=[False,True]).head(k)
            captured=float(selected.truth_Wh_per_m2_day.sum())
            records.append({'method':method,'top_k_wall_patches':k,
                'num_spatially_heldout_wall_patches':len(grouped),
                'reference_topk_reflection_exposure_Wh_per_m2_day_sum':upper,
                'captured_reference_exposure_Wh_per_m2_day_sum':captured,
                'percent_of_reference_topk_exposure':100*captured/upper if upper>0 else None,
                'topk_overlap_with_physics_reference':int(len(set(selected.face_id)&set(ideal.face_id))),
                'selected_face_ids':'|'.join(map(str,selected.face_id)),
                'score_units':'SUM of face-wise Wh/m²/day; equal-area patch ranking diagnostic, NOT total kWh',
            })
    metadata={'validation':'held-out against approximate physics only (NOT observations)',
        'target':'extra absorbed solar exposure caused by reflections on wall patches',
        'data_partition':'same locked date + spatial blocks as residual training report',
        'model_candidate_test_tuning':'none; report-only, no retraining',
        'comparator':'sort by no-reflection direct+sky exposure, a cheap but reflection-blind rule',
        'scope':'dark-roof scenario; wall patches only; one day; same NYC district',
        'counts_are_individual_correlated_patches_not_independent_roofs':True,
        'run_time_speedup_shown':False,
        'field_hotspot_detection_accuracy_proven':False,
        'source_no_reflection_cannot_predict_reflected_energy':True,
        'limitations':'This ranks modeled reflections, not hotspots measured by sensors; warm facade temperature and human heat exposure cannot be inferred.'}
    return grouped,pd.DataFrame(records),metadata


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--predictions',type=Path,default=ROOT/'outputs/ai_upgrade/reflection_residual/candidate_heldout_predictions.csv')
    p.add_argument('--out',type=Path,default=ROOT/'outputs/real_world_upgrade')
    a=p.parse_args();df=pd.read_csv(a.predictions)
    faces,bench,meta=score_hotspots(df)
    a.out.mkdir(parents=True,exist_ok=True)
    faces.to_csv(a.out/'heldout_wall_reflection_exposure_SIMULATED.csv',index=False)
    bench.to_csv(a.out/'locked_wall_hotspot_selection_SIMULATED.csv',index=False)
    meta['heldout_file_sha256']=hashlib.sha256(a.predictions.read_bytes()).hexdigest()
    (a.out/'hotspot_audit.json').write_text(json.dumps(meta,indent=2))
    print(bench[['method','top_k_wall_patches','percent_of_reference_topk_exposure','topk_overlap_with_physics_reference']].to_string(index=False))
    return bench

if __name__=='__main__':main()
