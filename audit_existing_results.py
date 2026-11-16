"""Inspection-only provenance audit of the exact original uploaded results ZIP.
Does not invent or replace unavailable observed Landsat data.
"""
from __future__ import annotations
import argparse,hashlib,io,json,zipfile
from pathlib import Path
import pandas as pd

def audit_results(path,out):
    p=Path(path)
    with zipfile.ZipFile(p) as z:
        names=set(z.namelist())
        am='historical_ml/audit.json';fm='historical_ml/physics_training_faces.csv'
        if am not in names or fm not in names:raise FileNotFoundError('Missing historical geometry audit or per-face thermal simulation')
        audit=json.loads(z.read(am))
        with z.open(fm) as fh:
            # Load only small columns to confirm date/time and actual model scenario.
            f=pd.read_csv(fh,usecols=['time_utc','face_id','scenario'])
        grouped=f.groupby(['time_utc','scenario']).size()
        if not (grouped==audit['face_count']).all():raise AssertionError('Incomplete model faces')
        real_obs_names=[n for n in names if 'landsat' in n.lower() and n.endswith('.csv')]
        result={'source_zip_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),
            'core_buildings':audit.get('core_buildings'),
            'model_faces':audit.get('face_count'),
            'source_geometry_sha256':audit.get('geometry_sha256'),
            'modeled_unique_hours':int(f.time_utc.nunique()),
            'modeled_first_utc':min(f.time_utc),'modeled_last_utc':max(f.time_utc),
            'scenario_names':sorted(f.scenario.unique().tolist()),
            'weather_provenance_verified_in_model_audit':audit.get('weather_provenance_verified'),
            'weather_file_present_inside_results_zip':audit.get('weather_file') in names,
            'observed_landsat_csv_members':real_obs_names,
            'independent_temperature_validation_completed':False,
            'reason':'No USGS Landsat surface-temperature observations in the supplied result archive' if not real_obs_names else 'External dataset requires quality and match review'}
    dest=Path(out);dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(result,indent=2))
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--results-zip',required=True)
    p.add_argument('--out',default='outputs/validation_v3/EXISTING_RESULTS_PREFLIGHT.json')
    a=p.parse_args();print(json.dumps(audit_results(a.results_zip,a.out),indent=2))
