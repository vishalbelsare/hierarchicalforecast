from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from datasetsforecast.hierarchical import HierarchicalData, HierarchicalInfo
from hierarchicalforecast.core import HierarchicalReconciliation, HierarchicalEvaluation
from hierarchicalforecast.methods import BottomUp, TopDown, MiddleOut, MinTrace, ERM
from statsforecast.core import StatsForecast
from statsforecast.models import naive, seasonal_naive

def mse(y, y_hat):
    return np.mean((y-y_hat)**2)

def get_reconcilers(group: str, tags_names: List[str]):
    reconcilers = [ 
        BottomUp(),
        MinTrace(method='ols'),
        MinTrace(method='wls_struct'),
        MinTrace(method='wls_var'),
        MinTrace(method='mint_shrink'),
        ERM(method='exact')
    ]
    if group != 'TourismLarge':
        reconcilers += [
            TopDown(method='forecast_proportions'),
            TopDown(method='average_proportions'),
            TopDown(method='proportion_averages'),
        ]
        for name in tags_names[1:-1]:
            reconcilers += [
                MiddleOut(level=name, top_down_method='forecast_proportions'),
                MiddleOut(level=name, top_down_method='average_proportions'),
                MiddleOut(level=name, top_down_method='proportion_averages'),
            ]
    return reconcilers

def get_models(seasonality: int):
    models = [
        naive,
        (seasonal_naive, seasonality)
    ]
    return models

def main():
    results_dir = Path('./results')
    for group, cls_group in HierarchicalInfo:
        results_group_dir = results_dir / group
        results_group_dir.mkdir(exist_ok=True, parents=True)
        # dataset and problem
        Y_df, S, tags = HierarchicalData.load('./data', group)
        Y_df['ds'] = pd.to_datetime(Y_df['ds'])
        h = cls_group.horizon
        freq = 'MS' if cls_group.freq == 'M' else cls_group.freq
        seasonality = cls_group.seasonality
        test_size = cls_group.test_size
        # Base forecasts
        base_fcsts_file = results_group_dir / 'base-forecasts.parquet'
        base_res_file = results_group_dir / 'base-residuals.parquet'
        if base_fcsts_file.is_file() and base_res_file.is_file():
            Y_h = pd.read_parquet(base_fcsts_file)
            Y_res = pd.read_parquet(base_res_file)
        else:
            models = get_models(seasonality)
            fcst = StatsForecast(Y_df.set_index('unique_id'), models=models, freq=freq, n_jobs=-1)
            Y_h = fcst.cross_validation(h, n_windows=None, test_size=test_size, residuals=True)
            Y_res = fcst.cross_validation_residuals()
            Y_h.to_parquet(base_fcsts_file)
            Y_res.to_parquet(base_res_file)
        # Reconciliation for each window
        reconcilers = get_reconcilers(group, list(tags.keys()))
        hrec = HierarchicalReconciliation(reconcilers=reconcilers)
        cutoffs = Y_h['cutoff'].unique()
        Y_h_rec = []
        for cutoff in cutoffs:
            Y_h_cutoff = Y_h.query('cutoff == @cutoff').drop(columns=['cutoff'])
            Y_res_cutoff = Y_res.query('cutoff == @cutoff').drop(columns=['cutoff'])
            Y_h_rec_cutoff = hrec.reconcile(Y_h_cutoff, Y_res_cutoff, S, tags)
            Y_h_rec_cutoff['cutoff'] = cutoff
            Y_h_rec.append(Y_h_rec_cutoff)
        Y_h_rec = pd.concat(Y_h_rec)
        Y_h_rec.to_parquet(results_group_dir / 'reconciled-forecasts.parquet')
        # Evaluation
        evaluator = HierarchicalEvaluation(evaluators=[mse])
        eval_ = evaluator.evaluate(Y_h_rec.drop(columns=['y', 'cutoff']), Y_h_rec[['ds', 'y']], tags, 'naive')
        eval_ = pd.melt(eval_, value_vars=eval_.columns.to_list(), var_name='model', value_name='loss', ignore_index=False)
        eval_[['model', 'rec_method']] = eval_['model'].str.split('/', expand=True, n=1)
        eval_['rec_method'] = eval_['rec_method'].fillna('NoReconciled')
        eval_ = eval_.set_index(['rec_method'], append=True)
        eval_ = eval_.reset_index()
        eval_ = eval_.pivot(index=['level', 'metric', 'rec_method'], columns='model', values='loss')
        eval_.to_parquet(results_group_dir / 'evaluation.parquet')
        print(eval_)

if __name__=="__main__":
    main()

