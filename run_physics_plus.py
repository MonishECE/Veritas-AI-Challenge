"""Real-NYC-geometry physics+ study. Not a verified urban-temperature forecast.

Run only with actually downloaded, consecutive Open-Meteo historical weather.
Outputs complete scientific provenance and optional per-face AI training corpus.
"""
from __future__ import annotations
import argparse, json, sys, hashlib
from time import perf_counter
from pathlib import Path
from shapely.geometry import shape
from dataclasses import replace
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
from integrated88 import load_verified_geometry,load_weather
from urbanray import build_scene
from physics_plus import (ASSUMED_MATERIALS,build_reciprocal_operator,materials_with_roof_alpha,
     sensitivity_materials,thermal_properties,solar_source_from_weather,shortwave_from_operator,
     step_two_layer)
from ai_inference import load_surrogate, infer_shortwave, ENHANCED_MODEL_FEATURES
from decision_engine import UrbanRayDecisionEngine
from geometry_features import build_geometry_graph


def scenario_materials(use_sensitivity=False):
    base={'dark_roof':materials_with_roof_alpha(.18),
          'reflective_roof':materials_with_roof_alpha(.58)}
    if use_sensitivity:
        base.update({
          'capacity_minus20_pct':sensitivity_materials(.8),
          'capacity_plus20_pct':sensitivity_materials(1.2),
          'emissivity_minus003':sensitivity_materials(1.0,-.03),
          'emissivity_plus003':sensitivity_materials(1.0,+.03),
          'conductivity_minus20_pct':sensitivity_materials(conductivity_factor=.8),
          'conductivity_plus20_pct':sensitivity_materials(conductivity_factor=1.2)})
    return base


def execute(weather_csv,output_dir,rays=8,shadow_samples=4,warmup_hours=48,
            hours=24,seed=44,sensitivity=False,save_faces=True,bounces=3,context_geojson=None,
            radiation_mode='physics',ai_model_path=None,hotspot_model_path=None):
    if radiation_mode not in ('physics','ai','compare','decision','compare_decision'):
        raise ValueError('radiation_mode must be physics, ai, compare, decision, or compare_decision')
    if radiation_mode in ('ai','compare','decision','compare_decision') and sensitivity:
        raise ValueError('Sensitivity mode varies properties outside the trained AI domain. Run physics mode.')
    bundle = None
    engine = None
    model_path = None
    specialist_path = None
    if radiation_mode in ('ai','compare','decision','compare_decision'):
        default_name = ('geometry_ai_upgrade/regression_surrogate.joblib'
                        if radiation_mode in ('decision','compare_decision')
                        else 'ai_upgrade/reflection_residual/selected_surrogate.joblib')
        model_path = Path(ai_model_path or ROOT/'outputs'/default_name)
        bundle = load_surrogate(model_path)
        if radiation_mode in ('decision','compare_decision'):
            specialist_path = Path(hotspot_model_path or
                ROOT/'outputs/geometry_ai_upgrade/hotspot_surrogate.joblib')
            engine = UrbanRayDecisionEngine(bundle,load_surrogate(specialist_path))
    src_geo=ROOT/'data'/'nyc_88_source.geojson' 
    records,sha=load_verified_geometry(src_geo)
    core_ids={r['id'] for r in records}
    context_file_hash=None
    if context_geojson is not None:
        context_file=Path(context_geojson)
        context_file_hash=hashlib.sha256(context_file.read_bytes()).hexdigest()
        context=json.loads(context_file.read_text())
        if context.get('type')!='FeatureCollection':raise ValueError('Context must be a GeoJSON FeatureCollection')
        extras=[]; extra_ids=set()
        for f in context['features']:
            pr=f['properties'];bid=str(pr['id'])
            if bid in core_ids or bid in extra_ids:continue
            if 'NYC Open Data' not in pr.get('source',''):raise ValueError('Context geometry must carry official NYC provenance')
            h=float(pr['height_m']);feet=float(pr['height_roof_ft'])
            if not np.isclose(h,feet*.3048,atol=1e-4) or not 3<=h<=450:
                raise ValueError('Invalid NYC context height conversion')
            geom=shape(f['geometry'])
            if not geom.is_valid or geom.is_empty:raise ValueError('Invalid NYC context footprint')
            extras.append({'id':bid,'height_m':h,'geometry':geom,'source':'NYC OTI verified API'})
            extra_ids.add(bid)
        print('Additional official NYC context features:',len(extras),flush=True)
        records=records+extras
    scene=build_scene(records,ground_cell_m=25,max_buildings=len(records))
    scene_ids=set(scene['building_id'])-{'GROUND'}
    if scene['source_is_synthetic'] or not core_ids.issubset(scene_ids):
        raise AssertionError('A source NYC core building was lost from the model')
    core_faces=np.array([(b in core_ids or b=='GROUND') for b in scene['building_id']])
    if not core_faces.any():raise ValueError('Empty core evaluation area')
    weather=load_weather(weather_csv)
    count=warmup_hours+hours
    if len(weather)<count:raise ValueError(f'Need {count} consecutive weather hours, only {len(weather)} supplied')
    weather=weather.iloc[:count].copy()
    if not (weather['timestamp'].diff().dropna()==pd.Timedelta(hours=1)).all():
        raise ValueError('weather must be continuous hourly observations')
    output=Path(output_dir);output.mkdir(parents=True,exist_ok=True)
    print(f'Building 88-building sparse view factors with {rays} rays per patch...',flush=True)
    P,sky,opmeta=build_reciprocal_operator(scene,rays=rays,seed=seed)
    # Build the local geometry graph once, not for every hour or scenario.
    geometry_graph = (build_geometry_graph(scene) if bundle is not None and
         list(bundle.get('features', []))==list(ENHANCED_MODEL_FEATURES) else None)
    n=len(scene['area'])
    kinds=scene['kind'];area=scene['area'];kinds_order=('roof','wall','ground')
    scenarios=scenario_materials(sensitivity)
    temp0=float(weather.iloc[0]['temperature_2m'])+273.15
    methods = {'physics':('physics',), 'ai':('ai',),
               'compare':('physics','ai','no_reflection'),
               'decision':('decision_ai',),
               'compare_decision':('physics','decision_ai','no_reflection')}[radiation_mode]
    temp_s={(name,method):np.full(n,temp0) for name in scenarios for method in methods}
    temp_d={(name,method):np.full(n,temp0) for name in scenarios for method in methods}
    properties = {name:thermal_properties(kinds,mat) for name,mat in scenarios.items()}
    refK=float(weather['temperature_2m'].iloc[:max(1,warmup_hours)].mean())+273.15
    results=[];faces=[]
    for i,row in weather.iterrows():
        dt=row['timestamp'].to_pydatetime()
        if (i%6==0):print(f'  weather hour {i+1}/{count}',flush=True)
        src,sun=solar_source_from_weather(scene,dt,
            max(0.,float(row['direct_normal_irradiance_instant'])),
            max(0.,float(row['diffuse_radiation_instant'])),sky,
            shadow_samples=shadow_samples,seed=seed+int(i))
        for name,materials in scenarios.items():
            props=properties[name]
            # Three-way comparison uses identical input geometry and sunlight.
            # Each thermal method has its own full warmup/history: NO temperature
            # states are copied from the full-physics reference.
            solar = {}
            info = {}
            runtime = {}
            if 'physics' in methods:
                begin=perf_counter()
                solar['physics'],info['physics'] = shortwave_from_operator(
                    src,P,sky,area,props['alpha'],bounces=bounces)
                runtime['physics']=(perf_counter()-begin)*1000
            if 'ai' in methods:
                begin=perf_counter()
                pred = infer_shortwave(bundle,scene,src,sky,sun,row,props['alpha'],
                                       geometry_graph=geometry_graph)
                solar['ai'],info['ai'] = pred.flux_Wm2,pred.metadata
                runtime['ai']=(perf_counter()-begin)*1000
            if 'decision_ai' in methods:
                begin=perf_counter()
                decision = engine.predict_scene(scene,src,sky,sun,row,props['alpha'],
                                                geometry_graph=geometry_graph)
                solar['decision_ai'] = decision.radiation.flux_Wm2
                info['decision_ai'] = decision.radiation.metadata | decision.metadata
                runtime['decision_ai']=(perf_counter()-begin)*1000
            if 'no_reflection' in methods:
                begin=perf_counter()
                solar['no_reflection'] = (src/area)*(1-props['alpha'])
                info['no_reflection'] = {'absorbed_W':float(np.dot(solar['no_reflection'],area)),
                    'initial_W':float(src.sum()),'energy_balance_verified':False}
                runtime['no_reflection']=(perf_counter()-begin)*1000
            for method in methods:
                sw=solar[method]
                swmeta=info[method]
                key=(name,method)
                Ts,Td,thermalmeta=step_two_layer(temp_s[key],temp_d[key],sw,
                    air_K=float(row['temperature_2m'])+273.15,
                    wind_ms=float(row['wind_speed_10m']),
                    rh_percent=float(row['relative_humidity_2m']),
                    cloud_fraction=float(row['cloud_cover'])/100,
                    P=P,sky=sky,scene=scene,materials=materials,
                    deep_reference_K=refK,dt=3600)
                temp_s[key],temp_d[key]=Ts,Td
                if i < warmup_hours:continue
                metrics={k:float(np.average(Ts[(kinds==k)&core_faces]-273.15,
                            weights=area[(kinds==k)&core_faces])) for k in kinds_order}
                if 'physics' in solar:
                    swerr=float(np.average(np.abs(sw-solar['physics']),weights=area))
                else:
                    swerr=float('nan')
                results.append({'time_utc':dt.isoformat(),'date_utc':dt.date().isoformat(),
                    'hour_in_evaluation':int(i-warmup_hours),
                    'scenario':name,'radiation_method':method,
                    'roof_alpha_assumed':materials['roof'].alpha,
                    'roof_C_simulated':metrics['roof'],
                    'wall_C_simulated':metrics['wall'],
                    'ground_C_simulated':metrics['ground'],
                    'total_shortwave_absorbed_MW':swmeta['absorbed_W']/1e6,
                    'SW_relative_budget_residual':swmeta.get('relative_residual',float('nan')),
                    'SW_unresolved_fraction':(swmeta['unresolved_W']/max(1,swmeta['initial_W']))
                        if 'unresolved_W' in swmeta else float('nan'),
                    'SW_area_weighted_MAE_against_physics_Wm2':swerr,
                    'SW_compute_ms':runtime[method],
                    'AI_energy_cap_applied':swmeta.get('source_cap_applied',False),
                    'AI_energy_cap_scale':swmeta.get('energy_cap_scale',1.0),
                    'decision_review_faces':int(decision.review_mask.sum()) if method=='decision_ai' else None,
                    'decision_mean_model_spread_Wm2':float(decision.model_disagreement_Wm2.mean()) if method=='decision_ai' else None,
                    'LW_net_MW':thermalmeta['LW_net_MW'],
                    'max_thermal_surface_residual_Wm2':thermalmeta['max_surface_residual_Wm2'],
                    'max_thermal_substrate_residual_Wm2':thermalmeta['max_deep_residual_Wm2'],
                    'air_temp_C_grid':float(row['temperature_2m'])})
                if save_faces:
                    face = pd.DataFrame({
                        'time_utc':dt.isoformat(),'date_utc':dt.date().isoformat(),
                        'scenario':name,'radiation_method':method,'face_id':np.arange(n),
                        'building_id':scene['building_id'],'kind':kinds,
                        'x_m':scene['center'][:,0],'y_m':scene['center'][:,1],
                        'z_m':scene['center'][:,2],'nx':scene['normal'][:,0],
                        'ny':scene['normal'][:,1],'nz':scene['normal'][:,2],
                        'area_m2':area,'albedo_assumed':props['alpha'],
                        'sun_east':sun[0],'sun_north':sun[1],'sun_up':sun[2],
                        'DNI_grid_Wm2':float(row['direct_normal_irradiance_instant']),
                        'DHI_grid_Wm2':float(row['diffuse_radiation_instant']),
                        'air_temp_C_grid':float(row['temperature_2m']),
                        'wind_grid_ms':float(row['wind_speed_10m']),
                        'sky_fraction_modeled':sky,'source_flux_Wm2':src/area,
                        'direct_plus_sky_abs_Wm2':src/area*(1-props['alpha']),
                        # This column stores the method's shortwave, not always
                        # the physical multibounce reference; see method label.
                        'method_abs_Wm2':sw,
                        'surface_temp_C_simulated':Ts-273.15})
                    if method=='decision_ai':
                        face['hotspot_specialist_abs_Wm2']=decision.hotspots.flux_Wm2
                        face['dual_model_disagreement_Wm2']=decision.model_disagreement_Wm2
                        face['decision_review_required']=decision.review_mask
                    faces.append(face)
    table=pd.DataFrame(results)
    table.to_csv(output/'thermal_hourly_summary.csv',index=False)
    if save_faces and faces:
        pd.concat(faces,ignore_index=True).to_csv(output/'physics_training_faces.csv',index=False)
    weather_prov_path=Path(weather_csv).with_suffix('.provenance.json')
    verified_download=False
    if weather_prov_path.exists():
        provenance=json.loads(weather_prov_path.read_text())
        verified_download=(provenance.get('csv_sha256')==hashlib.sha256(Path(weather_csv).read_bytes()).hexdigest()
                           and 'archive-api.open-meteo.com' in provenance.get('source_url',''))
    if not verified_download:
        print('WARNING: weather file has no matching archived-API provenance hash; cannot attest sourced weather',flush=True)
    meta={'simulation':'88 real NYC footprint polygon prisms, approximate physics, assumed materials',
          'core_buildings':88,'context_included':bool(context_geojson),
          'total_buildings_simulated':int(scene['building_count']),
          'face_count':int(n),'geometry_sha256':sha,'context_geojson_sha256':context_file_hash,
          'geometry_historical_limit':'2026 snapshot; not guaranteed correct for each weather date',
          'weather_file':str(Path(weather_csv).name),
          'weather_source':'Open-Meteo archive file with matching provenance' if verified_download else 'UNVERIFIED input (may be synthetic); NOT claimable as observed weather',
          'weather_provenance_verified':bool(verified_download),
          'weather_sha256':hashlib.sha256(Path(weather_csv).read_bytes()).hexdigest(),
          'warmup_hours':int(warmup_hours),'evaluated_hours':int(hours),
          'sun_location_latitude':float(sum(scene['bbox'][i] for i in [1,3])/2),
          'sun_location_longitude':float(sum(scene['bbox'][i] for i in [0,2])/2),
          'operator':opmeta,'shadow_samples':shadow_samples,'bounces':bounces,
          'materials_assumed':{k:vars(v) for k,v in ASSUMED_MATERIALS.items()},
          'scenario_names':list(scenarios),
          'radiation_method':radiation_mode,'compared_methods':list(methods),
          'surrogate_model_file':str(model_path.name) if model_path else None,
          'unified_decision_engine_enabled':engine is not None,
          'specialist_hotspot_file':str(specialist_path.name) if specialist_path else None,
          'decision_uses_two_original_unchanged_models':engine is not None,
          'dual_model_disagreement_is_calibrated_uncertainty':False,
          'surrogate_training_labels':'simulated multibounce SW absorption, not field measurements',
          'geometry_proxy_features':geometry_graph.params if geometry_graph is not None else None,
          'ai_energy_cap_does_not_verify_energy_conservation':True,
          'scenario_assumed_materials':{name:{k:vars(v) for k,v in mats.items()} for name,mats in scenarios.items()},
          'surface_temperature_accuracy':'NOT independently established',
          'water_evaporation_and_tree_canopies':'NOT included without verified geometry/material inputs'}
    (output/'audit.json').write_text(json.dumps(meta,indent=2))
    print('SAVED:',output, 'rows:',len(table),flush=True)
    return table,meta

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--weather-csv',required=True)
    p.add_argument('--output',default=str(ROOT/'outputs'/'physics_plus'))
    p.add_argument('--rays',type=int,default=8)
    p.add_argument('--shadow-samples',type=int,default=4)
    p.add_argument('--warmup-hours',type=int,default=48)
    p.add_argument('--hours',type=int,default=24)
    p.add_argument('--bounces',type=int,default=3)
    p.add_argument('--seed',type=int,default=44)
    p.add_argument('--sensitivity',action='store_true')
    p.add_argument('--no-face-csv',action='store_true')
    p.add_argument('--context-geojson',default=None)
    p.add_argument('--radiation-mode',choices=('physics','ai','compare','decision','compare_decision'),default='physics',
                   help='Physics reference, AI surrogate, or matched three-way experiment')
    p.add_argument('--ai-model',default=None,help='Trusted regression surrogate joblib artifact')
    p.add_argument('--hotspot-model',default=None,help='Trusted hotspot model; decision modes only')
    a=p.parse_args()
    execute(a.weather_csv,a.output,a.rays,a.shadow_samples,a.warmup_hours,a.hours,
            a.seed,a.sensitivity,not a.no_face_csv,a.bounces,a.context_geojson,
            a.radiation_mode,a.ai_model,a.hotspot_model)
