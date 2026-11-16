"""Compare direct ML vs AI-learned missing multi-bounce reflection correction.

Leakage controls: train/validation/test partitions are provided by the original
spatial block + date splitter; model family/hyperparameters selected on validation
only, then BOTH finalists are refit on train+validation once. No test tuning.
Targets are simulation outputs, never observed rooftop temperature.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error,r2_score
from train_surrogate import load_training,split,PHYSICS

TARGET='multibounce_abs_Wm2'
BASE='direct_plus_sky_abs_Wm2'
HYPERPARAMETERS=[{'max_leaf_nodes':15,'learning_rate':.07,'max_iter':150},
                 {'max_leaf_nodes':31,'learning_rate':.065,'max_iter':200},
                 {'max_leaf_nodes':20,'learning_rate':.05,'max_iter':250}]


def fit_experiment(train,val,test,out,seed=62):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    partition_sizes={'train':len(train),'validation':len(val),'test':len(test)}
    # Preserve identical sampling logic to original for fair comparison.
    if len(train)>45000:train=train.sample(n=45000,random_state=seed).copy()
    if len(val)>15000:val=val.sample(n=15000,random_state=seed+1).copy()
    def labels(df,mode):
        return (df[TARGET]-df[BASE]) if mode=='residual' else df[TARGET]
    def predict(model,df,mode):
        raw=model.predict(df[PHYSICS])
        if mode=='residual': return df[BASE].to_numpy()+np.maximum(0.,raw)
        return np.maximum(0.,raw)
    selection=[]
    for mode in ('direct','residual'):
        for params in HYPERPARAMETERS:
            mdl=HistGradientBoostingRegressor(**params,random_state=seed,
                   l2_regularization=1.0,min_samples_leaf=20)
            mdl.fit(train[PHYSICS],labels(train,mode))
            mae=mean_absolute_error(val[TARGET],predict(mdl,val,mode))
            selection.append({'mode':mode,'validation_MAE_Wm2':float(mae),**params})
    chosen={mode:sorted((x for x in selection if x['mode']==mode),
             key=lambda x:x['validation_MAE_Wm2'])[0] for mode in ('direct','residual')}
    # Choose based on validation only, NOT test.
    preferred=min(chosen,key=lambda mode:chosen[mode]['validation_MAE_Wm2'])
    fitted={};predictions={}
    combo=pd.concat([train,val],ignore_index=True)
    for mode in ('direct','residual'):
        par={k:v for k,v in chosen[mode].items() if k not in ('mode','validation_MAE_Wm2')}
        model=HistGradientBoostingRegressor(**par,random_state=seed,
                  l2_regularization=1.0,min_samples_leaf=20)
        model.fit(combo[PHYSICS],labels(combo,mode))
        fitted[mode]=model
        predictions[mode]=predict(model,test,mode)
        joblib.dump({'model':model,'features':list(PHYSICS),'target':TARGET,
            'prediction_mode':mode,'sklearn_version':sklearn.__version__,
            'note':'Simulated multi-bounce absorption targets; residual model learns missing reflections',
            'train_test_split':'strict date + spatial block','selection':'validation only'},
            out/f'{mode}_surrogate.joblib')
    selected_name=f'{preferred}_surrogate.joblib'
    # Separate shortcut to the chosen best-on-validation model for both pipelines.
    import shutil
    shutil.copyfile(out/selected_name,out/'selected_surrogate.joblib')
    y=test[TARGET].to_numpy()
    preds={
        'Mean-only':np.full(len(test),float(train[TARGET].mean())),
        'No-reflection physics':test[BASE].to_numpy(),
        'Direct AI':predictions['direct'],
        'Reflection-residual AI':predictions['residual'],
    }
    rows=[]
    for name,p in preds.items():
        rows.append({'model':name,'test_MAE_Wm2':float(mean_absolute_error(y,p)),
            'test_RMSE_Wm2':float(np.sqrt(mean_squared_error(y,p))),
            'test_R2':float(r2_score(y,p))})
    pd.DataFrame(rows).to_csv(out/'candidate_heldout_radiation_scores.csv',index=False)
    testfile=test[['time_utc','date_utc','scenario','building_id','kind','face_id','spatial_block']].copy()
    testfile['physics_label_Wm2']=y
    for name,p in preds.items():testfile['pred_'+name.replace(' ','_')]=p
    testfile.to_csv(out/'candidate_heldout_predictions.csv',index=False)
    audit={'partition_sizes_before_subsampling':partition_sizes,
           'train_rows_used':len(train),'validation_rows_used':len(val),
           'test_rows_never_used_for_model_selection':len(test),
           'hyperparameters_considered':selection,'best_by_mode':chosen,
           'selected_by_validation':preferred,'test_results':rows,
           'not_field_validated':True,'sklearn_version':sklearn.__version__}
    (out/'candidate_selection_audit.json').write_text(json.dumps(audit,indent=2))
    return pd.DataFrame(rows),audit


def main():
    p=argparse.ArgumentParser();p.add_argument('--faces',type=Path,required=True)
    p.add_argument('--out',type=Path,default=Path('outputs/ai_upgrade/reflection_residual'))
    a=p.parse_args()
    df=load_training(a.faces)
    train,val,test,split_meta=split(df)
    m,meta=fit_experiment(train,val,test,a.out)
    (a.out/'split_audit.json').write_text(json.dumps(split_meta,indent=2))
    print(m.to_string(index=False))
    print('Chosen by validation only:',meta['selected_by_validation'])

if __name__=='__main__':main()
