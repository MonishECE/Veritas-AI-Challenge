"""Build an OFFLINE, interactive, evidence-labelled UrbanRay AI decision dashboard.

Inputs: the supplied *real-geometry simulation* face CSV and a model retrained
on that CSV. Output: self-contained HTML + auditable metrics/recommendations.

Do not call any displayed model output a field-measured cooling result.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.inspection import permutation_importance
from plotly.offline import get_plotlyjs

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'src'))
from ai_inference import MODEL_FEATURES, ENHANCED_MODEL_FEATURES, load_surrogate, predict_from_frame
from decision_engine import UrbanRayDecisionEngine, screen_buildings, rank_wall_hotspots
from geometry_features import GEOMETRY_FEATURES, build_geometry_graph, table_scene
from integrated88 import load_verified_geometry
from urbanray import build_scene
from decision_benchmark import build_decision_audit
from reflection_hotspot_audit import score_hotspots
from observational_evidence import archived_landsat_evidence

SCENARIOS=('dark_roof','reflective_roof')
METHODS=('ai','physics','no_reflection')
TEST_DAY='2025-06-23'


def read_day_from_archive(source: Path, day: str) -> pd.DataFrame:
    # Only read the chosen original, untouched test date; no artificial weather.
    chunks=[]
    for chunk in pd.read_csv(source,chunksize=24000, dtype={'building_id':str}):
        selected=chunk.loc[(chunk['date_utc']==day)&chunk['scenario'].isin(SCENARIOS)]
        if len(selected): chunks.append(selected.copy())
    if not chunks: raise ValueError(f'No archived physics faces for {day}')
    df=pd.concat(chunks,ignore_index=True)
    # Exact one row per scenario/time/face, no gaps.
    if df.duplicated(['scenario','time_utc','face_id']).any():
        raise ValueError('Duplicated face IDs in archive')
    for (scenario,time),group in df.groupby(['scenario','time_utc']):
        ids=np.sort(group.face_id.to_numpy())
        if not np.array_equal(ids,np.arange(len(ids))):
            raise ValueError(f'Incomplete mesh for {scenario}, {time}')
    return df


def model_day_predictions(df:pd.DataFrame,bundle:dict, engine:UrbanRayDecisionEngine|None=None) -> tuple[pd.DataFrame,dict]:
    out=[];inference_ms=0.;limit_count=0;review_count=0
    df=df.copy();df['kind_code']=df['kind'].map({'roof':0,'wall':1,'ground':2}).astype(int)
    enriched=list(bundle.get('features', MODEL_FEATURES))==list(ENHANCED_MODEL_FEATURES)
    # Exactly one graph across all hours; avoids repeated tree construction.
    graph=build_geometry_graph(table_scene(df.drop_duplicates('face_id'))) if enriched else None
    for (scenario,time),group in df.groupby(['scenario','time_utc'],sort=True):
        group=group.sort_values('face_id').copy()
        src=group.source_flux_Wm2.to_numpy()*group.area_m2.to_numpy()
        if enriched:
            proxy=graph.features(group.source_flux_Wm2.to_numpy(),group.albedo_assumed.to_numpy())
            for i,feature in enumerate(GEOMETRY_FEATURES):group[feature]=proxy[:,i]
        shared_frame=group.loc[:,bundle.get('features',MODEL_FEATURES)]
        if engine is not None:
            result=engine.predict_frame(shared_frame,src,group.area_m2.to_numpy())
            pred=result.radiation
            group['hotspot_abs_Wm2']=result.hotspots.flux_Wm2
            group['model_disagreement_Wm2']=result.model_disagreement_Wm2
            group['engine_review_mask']=result.review_mask
            review_count += int(result.review_mask.sum())
        else:
            pred=predict_from_frame(bundle,shared_frame,src,group.area_m2.to_numpy())
        group['ai_abs_Wm2']=pred.flux_Wm2
        group['baseline_abs_Wm2']=group.direct_plus_sky_abs_Wm2
        group['physics_abs_Wm2']=group.multibounce_abs_Wm2
        inference_ms += pred.metadata['inference_ms_excludes_feature_construction']
        limit_count += int(pred.metadata['source_cap_applied'])
        out.append(group)
    values=pd.concat(out,ignore_index=True)
    return values,{'source_cap_triggered_cases':limit_count,
        'surrogate_inference_total_ms_across_day':round(inference_ms,2),
        'inference_timing_excludes_csv_read_and_feature_construction':True,
        'unified_engine_enabled':engine is not None,
        'engine_model_disagreement_review_face_hours':review_count}


def evaluate_heldout(pred_file:Path) -> tuple[pd.DataFrame,pd.DataFrame]:
    df=pd.read_csv(pred_file)
    model_column=('pred_Reflection-residual_AI' if 'pred_Reflection-residual_AI' in df
                   else 'pred_Physics-informed_ML')
    names={'ai':model_column,'no_reflection':'pred_No-reflection_physics',
           'mean_only':'pred_Mean-only'}
    if 'pred_Direct_AI' in df:
        names['direct_ai']='pred_Direct_AI'
    rows=[]
    for kind,part in [('all',df)]+list(df.groupby('kind',sort=True)):
        for method,col in names.items():
            y=part.physics_label_Wm2.to_numpy()
            p=part[col].to_numpy()
            rows.append({'kind':kind,'method':method,'n_patches':len(part),
                'MAE_Wm2':float(np.mean(np.abs(y-p))),
                'RMSE_Wm2':float(np.sqrt(np.mean((y-p)**2))),
                'absolute_error_p90_Wm2':float(np.percentile(np.abs(y-p),90))})
    return pd.DataFrame(rows),df


def importance_from_heldout(df:pd.DataFrame,testpred:pd.DataFrame,bundle:dict,
                            sample_n:int=2500) -> pd.DataFrame:
    # Strict locked spatial + date test rows; no training/selection here.
    keys=testpred[['time_utc','scenario','face_id']].drop_duplicates()
    if keys.duplicated(['time_utc','scenario','face_id']).any():
        raise ValueError('Test keys not unique')
    joined=df.merge(keys,on=['time_utc','scenario','face_id'],how='inner',validate='one_to_one')
    if len(joined)!=len(testpred):
        raise ValueError(f'Test join was incomplete: {len(joined)} != {len(testpred)}')
    rng=np.random.default_rng(179)
    if len(joined)>sample_n:
        joined=joined.iloc[rng.choice(len(joined),size=sample_n,replace=False)]
    features=list(bundle.get('features',MODEL_FEATURES))
    data=joined.loc[:,features]
    is_residual=bundle.get('prediction_mode','direct')=='residual'
    # Evaluate ONLY the correction learned by AI when residual is selected.
    # Keep the physical direct+sky baseline fixed instead of permuting that
    # baseline into the scorer (which would exaggerate the AI's importance).
    target=(joined.multibounce_abs_Wm2-joined.direct_plus_sky_abs_Wm2
            if is_residual else joined.multibounce_abs_Wm2)
    def effective_model_score(estimator,X,y):
        p=estimator.predict(X)
        p=np.maximum(0.,p)
        return -float(np.mean(np.abs(np.asarray(y)-p)))
    score=permutation_importance(bundle['model'],data,target,
         scoring=effective_model_score,n_repeats=3,random_state=179,n_jobs=1)
    result=pd.DataFrame({'feature':features,
        'mean_extra_MAE_Wm2_if_permuted':score.importances_mean,
        'std_extra_MAE_Wm2':score.importances_std})
    return result.sort_values('mean_extra_MAE_Wm2_if_permuted',ascending=False).reset_index(drop=True)


def rank_buildings(df:pd.DataFrame) -> tuple[pd.DataFrame,dict]:
    roof=df[df.kind=='roof'].copy()
    # Sunlit time only, same time set across both scenarios.
    roof=roof[(roof.DNI_grid_Wm2+roof.DHI_grid_Wm2>0)&(roof.sun_up>0)]
    if roof.empty: raise ValueError('No daylight hours found')
    # One roof area per building; weighted across all recorded daylight patches/hours.
    group=roof.groupby(['building_id','scenario'],sort=True)
    rows=group.apply(lambda g:pd.Series({
        'ai_Wm2':np.average(g.ai_abs_Wm2,weights=g.area_m2),
        'physics_Wm2':np.average(g.physics_abs_Wm2,weights=g.area_m2),
        'temp_physics_C':np.average(g.surface_temp_C_simulated,weights=g.area_m2),
        'area_m2':g.groupby('face_id').area_m2.first().sum(),
    }),include_groups=False).reset_index()
    pivot=rows.pivot(index='building_id',columns='scenario',values=['ai_Wm2','physics_Wm2','temp_physics_C','area_m2'])
    if pivot.isna().any().any():raise ValueError('A building is missing a roof scenario')
    results=pd.DataFrame({'building_id':pivot.index.astype(str),
        'roof_area_m2':pivot['area_m2']['dark_roof'].to_numpy(),
        'ai_dark_Wm2':pivot['ai_Wm2']['dark_roof'].to_numpy(),
        'ai_reflective_Wm2':pivot['ai_Wm2']['reflective_roof'].to_numpy(),
        'physics_dark_Wm2':pivot['physics_Wm2']['dark_roof'].to_numpy(),
        'physics_reflective_Wm2':pivot['physics_Wm2']['reflective_roof'].to_numpy(),
        'physics_dark_temp_C':pivot['temp_physics_C']['dark_roof'].to_numpy(),
        'physics_reflective_temp_C':pivot['temp_physics_C']['reflective_roof'].to_numpy()})
    # City action metric: solar HEAT ABSORPTION avoided on this model day.
    # W/m² * m² * 1 simulated hour / 1000 = kWh of solar heat, NOT electrical kWh.
    # For each roof, include all 24 hours rather than extrapolating daylight averages.
    roof_full=df[df.kind=='roof'].copy()
    daily=roof_full.groupby(['building_id','scenario']).apply(lambda g:pd.Series({
        'ai_heat_kWh':float(np.sum(g.ai_abs_Wm2*g.area_m2)/1000.),
        'physics_heat_kWh':float(np.sum(g.physics_abs_Wm2*g.area_m2)/1000.),
        'no_reflection_heat_kWh':float(np.sum(g.baseline_abs_Wm2*g.area_m2)/1000.),
    }),include_groups=False).unstack('scenario')
    for method in ('ai','physics','no_reflection'):
        gain=daily[f'{method}_heat_kWh']['dark_roof']-daily[f'{method}_heat_kWh']['reflective_roof']
        results[f'{method}_solar_heat_avoided_kWh_simulated_day']=results.building_id.map(gain)
    if results.filter(like='_solar_heat_avoided_kWh_simulated_day').isna().any().any():
        raise ValueError('A building is missing daily solar heat counterfactuals')
    results['ai_estimated_roof_solar_reduction_Wm2']=results.ai_dark_Wm2-results.ai_reflective_Wm2
    results['physics_roof_solar_reduction_Wm2']=results.physics_dark_Wm2-results.physics_reflective_Wm2
    results['physics_simulated_roof_temperature_delta_C']=results.physics_dark_temp_C-results.physics_reflective_temp_C
    results=results.sort_values('ai_estimated_roof_solar_reduction_Wm2',ascending=False).reset_index(drop=True)
    results.insert(0,'solar_reduction_rank',np.arange(1,len(results)+1))
    rho,p=spearmanr(results.ai_estimated_roof_solar_reduction_Wm2,results.physics_roof_solar_reduction_Wm2)
    stats={'rankings_compared_to':'same-date multibounce physics simulator, NOT measured cooling',
        'building_count_ranked':len(results),
        'ai_vs_physics_building_solar_saving_rank_spearman':float(rho),
        'ai_vs_physics_rank_correlation_p_value_descriptive_only':float(p),
        'ranking_daylight_average':'2025-06-23 sampled gridded weather, daylight hours; modeled roof patches'}
    return results,stats


def validate_geometry(df:pd.DataFrame) -> dict:
    records,_=load_verified_geometry(ROOT/'data'/'nyc_88_source.geojson')
    scene=build_scene(records,ground_cell_m=25,max_buildings=88)
    first=df[(df.scenario=='dark_roof')&(df.time_utc==sorted(df.time_utc.unique())[0])].sort_values('face_id')
    if len(scene['area'])!=len(first):
        raise ValueError('Mesh face count differs from original archived faces')
    centers=np.column_stack([first.x_m,first.y_m,first.z_m])
    if not np.allclose(scene['center'],centers,atol=.005):
        raise ValueError('Current geometry differs from archived source face centers')
    if not np.allclose(scene['area'],first.area_m2.to_numpy(),rtol=1e-5,atol=1e-3):
        raise ValueError('Mesh patch areas do not match archive')
    return scene


def compact_js_data(df:pd.DataFrame, scene:dict, rankings:pd.DataFrame) -> dict:
    times=sorted(df.time_utc.unique())
    bid_ids=rankings.building_id.tolist()
    # Geometry is read once from verified source; every face ID matched above.
    v=scene['triangles'].reshape((-1,3))
    ind=np.arange(len(v)).reshape((-1,3))
    roof=scene['kind']=='roof'
    marks=[]
    for bid in bid_ids:
        ix=(scene['building_id']==bid)&roof
        w=scene['area'][ix]
        if not ix.any():raise ValueError('Building missing roof face: '+bid)
        ctr=np.average(scene['center'][ix],axis=0,weights=w)
        marks.append({'id':bid,'x':round(float(ctr[0]),2),'y':round(float(ctr[1]),2),'z':round(float(ctr[2]+3),2)})
    states={}; daily={};
    for scenario in SCENARIOS:
        group=df[df.scenario==scenario]
        states[scenario]={}
        for time in times:
            g=group[group.time_utc==time].sort_values('face_id')
            states[scenario][time]={
                'ai':np.round(g.ai_abs_Wm2.to_numpy(),2).tolist(),
                'physics':np.round(g.physics_abs_Wm2.to_numpy(),2).tolist(),
                'no_reflection':np.round(g.baseline_abs_Wm2.to_numpy(),2).tolist(),
            }
            if 'hotspot_abs_Wm2' in g:
                states[scenario][time]['hotspot_ai']=np.round(g.hotspot_abs_Wm2.to_numpy(),2).tolist()
        # Series for the selected roof, at each hour. Area-weighted per roof.
        each=group[group.kind=='roof'].groupby(['building_id','time_utc'],sort=False)
        means=each.apply(lambda g:pd.Series({
            'ai':np.average(g.ai_abs_Wm2,weights=g.area_m2),
            'physics':np.average(g.physics_abs_Wm2,weights=g.area_m2),
            'no_reflection':np.average(g.baseline_abs_Wm2,weights=g.area_m2),
            'physics_temp_C':np.average(g.surface_temp_C_simulated,weights=g.area_m2),
        }),include_groups=False).reset_index()
        for bid,g in means.groupby('building_id'):
            daily.setdefault(str(bid),{})[scenario]={
              name:np.round(g.set_index('time_utc').reindex(times)[name].to_numpy(),2).tolist()
              for name in ['ai','physics','no_reflection','physics_temp_C']}
    ri=np.where(roof)[0]
    return {'times':times,'mesh':{
           'x':np.round(v[:,0],2).tolist(),'y':np.round(v[:,1],2).tolist(),
           'z':np.round(v[:,2],2).tolist(),
           'i':ind[:,0].tolist(),'j':ind[:,1].tolist(),'k':ind[:,2].tolist(),
           'roof_i':ri.tolist(),
           'roof_x':np.round(scene['center'][ri,0],2).tolist(),
           'roof_y':np.round(scene['center'][ri,1],2).tolist()},
           'centroids':marks,'states':states,'building_series':daily,
           'rankings':json.loads(rankings.round(3).to_json(orient='records')),
           'default_time_index':min(range(len(times)),key=lambda ix:abs(int(times[ix][11:13])-17))}


def create_dashboard(path:Path,data:dict,meta:dict,held:pd.DataFrame,importance:pd.DataFrame,evidence:dict):
    # All resources including Plotly.js embedded: no network or account needed.
    jsdata=json.dumps(data,separators=(',',':'),allow_nan=False)
    jsmeta=json.dumps(meta,separators=(',',':'),allow_nan=False)
    benchmark=held[held.kind=='all'].set_index('method')
    model_mae=benchmark.loc['ai','MAE_Wm2'];base_mae=benchmark.loc['no_reflection','MAE_Wm2']
    improvement=(1-model_mae/base_mae)*100
    imp_rows=json.dumps(importance.head(12).round(3).to_dict('records'),separators=(',',':'))
    html=(ROOT/'dashboard_template.html').read_text(encoding='utf-8')
    html=(html.replace('__PLOTLY__',get_plotlyjs()).replace('__DATA__',jsdata)
        .replace('__META__',jsmeta).replace('__IMPORTANCE__',imp_rows)
        .replace('__EVIDENCE__',json.dumps(evidence,separators=(',',':'),allow_nan=False))
        .replace('__VALIDATION_2025__',json.dumps(json.loads((ROOT/'VALIDATION_2025_SUMMARY.json').read_text()),separators=(',',':'))))
    path.write_text(html,encoding='utf-8')
    return {'html_bytes':path.stat().st_size,'heldout_AI_MAE_Wm2':float(model_mae),
            'improvement_percent':float(improvement)}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--faces',type=Path,default=ROOT/'outputs/historical_ml/physics_training_faces.csv')
    parser.add_argument('--model',type=Path,default=ROOT/'outputs/ai_upgrade/reflection_residual/selected_surrogate.joblib')
    parser.add_argument('--hotspot-model',type=Path,default=None,
                        help='Keep regression model unchanged; add trained specialist via unified decision engine')
    parser.add_argument('--heldout',type=Path,default=ROOT/'outputs/ai_upgrade/reflection_residual/candidate_heldout_predictions.csv')
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/ai_upgrade')
    a=parser.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    start=perf_counter()
    print('Reading archived physics simulations for locked test date...',flush=True)
    df=read_day_from_archive(a.faces,TEST_DAY)
    bundle=load_surrogate(a.model)
    engine=(UrbanRayDecisionEngine(bundle,load_surrogate(a.hotspot_model))
            if a.hotspot_model is not None else None)
    scene=validate_geometry(df)
    print('Running the shared-feature unified decision engine over both roof scenarios...'
          if engine else 'Running re-trained AI over original 88-building scenarios...',flush=True)
    df,infmeta=model_day_predictions(df,bundle,engine=engine)
    held,testpred=evaluate_heldout(a.heldout)
    # Recompute every decision benchmark from the untouched held-out prediction
    # file on each build. These evaluate simulator agreement, not field outcome.
    savings, topk, decision_meta=build_decision_audit(testpred,scene['area'])
    hotspot_faces, hotspot_topk, hotspot_meta=score_hotspots(testpred)
    evidence_dir=a.output.parent/'real_world_upgrade'
    evidence_dir.mkdir(parents=True,exist_ok=True)
    savings.to_csv(evidence_dir/'heldout_roof_portion_solar_heat_savings_SIMULATED.csv',index=False)
    topk.to_csv(evidence_dir/'locked_decision_topk_benchmark_SIMULATED.csv',index=False)
    hotspot_faces.to_csv(evidence_dir/'heldout_wall_reflection_exposure_SIMULATED.csv',index=False)
    hotspot_topk.to_csv(evidence_dir/'locked_wall_hotspot_selection_SIMULATED.csv',index=False)
    (evidence_dir/'decision_audit.json').write_text(json.dumps(decision_meta,indent=2))
    (evidence_dir/'hotspot_audit.json').write_text(json.dumps(hotspot_meta,indent=2))
    observed=archived_landsat_evidence()
    (evidence_dir/'observational_evidence_PHYSICS_ONLY.json').write_text(json.dumps(observed,indent=2))
    wall_hot=hotspot_topk[hotspot_topk.top_k_wall_patches.isin((10,20,50))]
    roof_dec=topk[topk.top_k_roof_portions.isin((3,5,10))]
    upgraded_report_path=a.model.parent/'model_audit.json'
    upgraded_report=json.loads(upgraded_report_path.read_text()) if upgraded_report_path.exists() else {}
    evidence={'model_comparison':upgraded_report.get('results_on_previously_inspected_test',[]),
              'old_test_reused':bool(upgraded_report),
              'wall_hotspots':json.loads(wall_hot.round(4).to_json(orient='records')),
              'roof_selections':json.loads(roof_dec.round(4).to_json(orient='records')),
              'observed':observed,
              'no_Ai_observational_validation_done':True} 
    print('Computing locked-heldout permutation feature importance...',flush=True)
    importance=importance_from_heldout(df,testpred,bundle)
    rankings,ranking_stats=rank_buildings(df)
    if engine is not None:
        # These two datasets come from the same dual-model inference run; there
        # is no hand-coded browser-side approximation of either model.
        screened,screen_meta=screen_buildings(df,scene)
        hotspot_priority=rank_wall_hotspots(df,top_k=20)
        screened.to_csv(a.output/'unified_building_screen_SIMULATED.csv',index=False)
        hotspot_priority.to_csv(a.output/'unified_wall_hotspots_SIMULATED.csv',index=False)
        engine_evidence={**screen_meta,
            'screenings':json.loads(screened.round(5).to_json(orient='records')),
            'top_hotspots':json.loads(hotspot_priority.round(4).to_json(orient='records')),
            'regression_model_sha256':hashlib.sha256(a.model.read_bytes()).hexdigest(),
            'hotspot_model_sha256':hashlib.sha256(a.hotspot_model.read_bytes()).hexdigest(),
            'model_weights_modified':False,
            'model_disagreement_review_face_hours':infmeta['engine_model_disagreement_review_face_hours']}
        (a.output/'unified_decision_audit.json').write_text(json.dumps(engine_evidence,indent=2,allow_nan=False))
        evidence['unified_engine']={k:v for k,v in engine_evidence.items() if k not in ('screenings','top_hotspots')}
    else:
        engine_evidence=None
    held.to_csv(a.output/'ai_heldout_diagnostics.csv',index=False)
    importance.to_csv(a.output/'ai_heldout_permutation_importance.csv',index=False)
    rankings.to_csv(a.output/'ai_cooling_rankings_SIMULATED.csv',index=False)
    demo=compact_js_data(df,scene,rankings)
    if engine_evidence is not None:
        demo['unified_engine']={
            'screenings':engine_evidence['screenings'],
            'top_hotspots':engine_evidence['top_hotspots'],
            'review_face_hours':engine_evidence['model_disagreement_review_face_hours'],
            'model_weights_modified':False,
            'validated_against_field_measurements':False,
            'wall_scenario_change_is_individual_roof_causal_effect':False}
    base=held[(held.kind=='all')&(held.method=='no_reflection')].iloc[0]
    ai=held[(held.kind=='all')&(held.method=='ai')].iloc[0]
    meta={**ranking_stats,**infmeta,'source_date':TEST_DAY,
        'heldout_ai_MAE_Wm2':float(ai.MAE_Wm2),
        'heldout_baseline_MAE_Wm2':float(base.MAE_Wm2),
        'heldout_percent_improvement_vs_no_reflection':100*(1-ai.MAE_Wm2/base.MAE_Wm2),
        'selected_ai_prediction_mode':bundle.get('prediction_mode','direct'),
        'unified_engine_enabled':engine is not None,
        'hotspot_model_file':a.hotspot_model.name if a.hotspot_model is not None else None,
        'heldout_roof_absolute_error_p90_Wm2':float(held[(held.kind=='roof')&(held.method=='ai')].iloc[0].absolute_error_p90_Wm2),
        'test_rows':int(ai.n_patches),'strict_heldout_data':'spatial blocks on 2025-06-23, previously inspected in older versions',
        'old_test_reused_not_new_independent_lockbox':True,
        'daily_demo_data':'full test date; not identical to strict test partition',
        'real_building_geometry':True,'simulated_solar_labels':True,
        'real_world_roof_temperature_accuracy_proven':False,
        'cooling_effect_measured_in_field':False,
        'weather_file_for_original_simulation_in_package':False,
        'model_path':str(a.model.relative_to(ROOT)) if a.model.is_relative_to(ROOT) else str(a.model),
        'full_original_faces_sha256':hashlib.sha256(a.faces.read_bytes()).hexdigest(),
        'dashboard_generation_seconds':round(perf_counter()-start,2)}
    (a.output/'ai_dashboard_audit.json').write_text(json.dumps(meta,indent=2))
    print('Generating self-contained interactive HTML...',flush=True)
    generated=create_dashboard(a.output/'UrbanRay_AI_Explorer_OFFLINE.html',demo,meta,held,importance,evidence)
    print(json.dumps({'dashboard':generated,'audit':meta},indent=2),flush=True)

if __name__=='__main__':main()
