"""Paired AI/physics/no-reflection comparison to actual Landsat observations.

A strict, no-calibration, scene-level EXPLORATORY diagnostic of a verified
REAL-WEATHER three-method simulation. Not field roof-temperature validation:
Landsat TIRS is spatially mixed; the model output is a bolometric proxy.

This program deliberately refuses synthetic or unverified weather. It cannot
produce scores merely from the archived 2025 AI shortwave predictions. Users
must supply a NEW, time-matched real-weather thermal simulation for all 3
radiation paths; the archive does NOT contain one.
"""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
from validate_landsat_pixels import run as run_matcher

METHODS=('physics','ai','no_reflection')


def check_comparison_audit(meta: dict) -> None:
    if meta.get('radiation_method') != 'compare' or set(meta.get('compared_methods',[])) != set(METHODS):
        raise ValueError('Requires a three-way physics/AI/no-reflection comparison run')
    if meta.get('weather_provenance_verified') is not True:
        raise ValueError('UNVERIFIED weather: no observational comparison is permitted')
    if 'Open-Meteo' not in str(meta.get('weather_source','')):
        raise ValueError('Requires genuine Open-Meteo archive weather provenance')
    if meta.get('context_included'):
        raise ValueError('The current Landsat matcher only supports the original 88-building core geometry')
    if int(meta.get('warmup_hours',0))<24:
        raise ValueError('Temperature model requires >=24h warmup for this diagnostic')
    if int(meta.get('evaluated_hours',0))<24:
        raise ValueError('Requires at least 24 matching evaluation hours')
    if str(meta.get('weather_file','')).upper().find('SMOKE_ONLY')>=0:
        raise ValueError('Synthetic weather fixture is not valid for observational comparison')


def paired_scene_metrics(scene_rows: dict[str,pd.DataFrame]) -> tuple[pd.DataFrame,dict]:
    # Match exact overpasses; never compare one model on different Landsat scenes.
    combined=[]
    for method in METHODS:
        d=scene_rows[method]
        required={'scene_id','acquisition_utc','observed_scene_mean_C','modeled_scene_mean_proxy_C'}
        if not required.issubset(d.columns):
            raise ValueError(f'Incomplete {method} scene diagnostic')
        if d.duplicated(['scene_id','acquisition_utc']).any():
            raise ValueError('Duplicate satellite scene/time')
        d=d[list(required)].rename(columns={'modeled_scene_mean_proxy_C':f'{method}_proxy_C',
                                          'observed_scene_mean_C':f'obs_{method}_C'})
        combined.append(d)
    match=combined[0]
    for d in combined[1:]:
        match=match.merge(d,on=['scene_id','acquisition_utc'],how='inner',validate='one_to_one')
    # All methods should use the same observed scene values, not a changing test sample.
    for method in METHODS[1:]:
        if not np.allclose(match[f'obs_{method}_C'],match['obs_physics_C'],atol=1e-9):
            raise ValueError('Observed satellite scene values differ across methods')
    if len(match)==0:
        return match,{'status':'NO_COMMON_SATELLITE_MATCHES','paired_scenes':0,
           'ai_observational_accuracy_established':False}
    match=match.rename(columns={'obs_physics_C':'observed_landsat_scene_mean_C'})
    match=match.drop(columns=[f'obs_{m}_C' for m in METHODS[1:]])
    # Treat dates, not grid cells, as the descriptive units. No post-hoc fitting.
    summaries={}
    for method in METHODS:
        err=match[f'{method}_proxy_C'].to_numpy()-match.observed_landsat_scene_mean_C.to_numpy()
        summaries[method]={'scene_mean_proxy_MAE_C':float(np.mean(np.abs(err))),
                           'scene_mean_proxy_RMSE_C':float(np.sqrt(np.mean(err*err))),
                           'scene_mean_proxy_bias_C':float(np.mean(err))}
    report={'status':'EXPLORATORY_PAIRED_SCENE_PROXY_DIAGNOSTIC_NOT_ROOF_VALIDATION',
        'paired_scenes':len(match),
        'scene_mean_results':summaries,
        'uses_real_Landsat_ST':True,
        'weather_is_gridded_reanalysis_not_roof_sensor':True,
        'ai_observational_roof_accuracy_established':False,
        'AI_improves_against_satellite_proxy': bool(summaries['ai']['scene_mean_proxy_MAE_C']<summaries['no_reflection']['scene_mean_proxy_MAE_C']),
        'no_fit_or_calibration_using_evaluation_scenes':True,
        'important_limitations':['Proxy emission is not Landsat thermal inversion',
            'Native TIRS support is much wider than its resampled 30m output grid',
            'All three models share thermal assumptions and model forcing',
            'Existing roofs may not match assumed dark-roof albedo',
            'No measured cool-roof intervention outcomes',
            'No claim of real roof-temperature predictive accuracy']}
    return match,report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--faces',type=Path,required=True,help='physics_training_faces.csv from REAL-WEATHER --radiation-mode compare run')
    p.add_argument('--pixels',type=Path,required=True,help='actual USGS Landsat EE export, NOT test fixture')
    p.add_argument('--out',type=Path,default=Path('outputs/real_world_upgrade/new_observational_test'))
    a=p.parse_args()
    auditpath=a.faces.parent/'audit.json'
    if not auditpath.is_file():raise ValueError('Missing simulation audit.json next to faces')
    audit=json.loads(auditpath.read_text())
    check_comparison_audit(audit)
    if not a.pixels.is_file() or not a.faces.is_file():raise FileNotFoundError('Missing real source data')
    a.out.mkdir(parents=True,exist_ok=True)
    scenes={}
    for method in METHODS:
        directory=a.out/f'{method}_EXPLORATORY'
        result=run_matcher(a.pixels,faces_csv=a.faces,out=directory,method=method)
        scene_csv=directory/'scene_level_EXPLORATORY.csv'
        scenes[method]=pd.read_csv(scene_csv)
        print(method, result['usable_scene_count'],'scene(s)')
    merged,report=paired_scene_metrics(scenes)
    merged.to_csv(a.out/'paired_Landsat_scene_predictions_EXPLORATORY.csv',index=False)
    report['faces_sha256']=hashlib.sha256(a.faces.read_bytes()).hexdigest()
    report['satellite_file_sha256']=hashlib.sha256(a.pixels.read_bytes()).hexdigest()
    report['simulation_weather_sha256']=audit['weather_sha256']
    (a.out/'PAIRED_VALIDATION_STATUS.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
