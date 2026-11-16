"""Conservative display metadata for ALREADY archived physics-only Landsat comparisons.

Do NOT reuse this historical physics comparison as a measured AI result.
All Landsat scenes are spatially mixed thermal retrievals and model output is
an approximate bolometric emission proxy, not independently observed roof T.
"""
from __future__ import annotations
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
SCENES=ROOT/'outputs/multidate_validation/scene_level_EXPLORATORY.csv'


def archived_landsat_evidence(path: Path=SCENES) -> dict:
    df=pd.read_csv(path)
    required={'scene_id','acquisition_utc','observed_scene_mean_C',
              'modeled_scene_mean_proxy_C','split_preregistered','matched_30m_grid_cells'}
    if not required.issubset(df.columns):
        raise ValueError(f'Missing observational scene fields {required-set(df.columns)}')
    if df.duplicated(['scene_id','acquisition_utc']).any():
        raise ValueError('Duplicate scene/time in observational archive')
    rows=[]; stats={}
    for split in ('development','heldout'):
        sub=df[df.split_preregistered==split]
        if len(sub)==0:raise ValueError(f'Missing {split} Landsat comparison split')
        err=sub.modeled_scene_mean_proxy_C.to_numpy()-sub.observed_scene_mean_C.to_numpy()
        stats[split]={'scenes':int(len(sub)),'raw_scene_proxy_MAE_C':round(float(np.abs(err).mean()),3),
                      'raw_scene_proxy_RMSE_C':round(float(np.sqrt(np.mean(err*err))),3)}
        for _,r in sub.sort_values('acquisition_utc').iterrows():
            rows.append({'date_utc':r.acquisition_utc[:10], 'split':split,
                         'observed_landsat_LST_C':round(float(r.observed_scene_mean_C),3),
                         'physics_modeled_emission_proxy_C':round(float(r.modeled_scene_mean_proxy_C),3),
                         'thermal_grid_cells_correlated':int(r.matched_30m_grid_cells)})
    return {'status':'ARCHIVED_PHYSICS_ONLY_EXPLORATORY_LANDSAT_COMPARISON',
            'raw_file_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'development':stats['development'],'heldout':stats['heldout'],
            'scenes':rows,'AI_measured_temperature_validation_completed':False,
            'heldout_scene_count':stats['heldout']['scenes'],
            'observed_data_is_Landsat_ST':True,
            'limitation':'Older PHYSICS-ONLY proxy vs 100m-native Landsat ST resampled to 30m. No AI thermal validation, no real roof thermometry or measured cooling.'}
