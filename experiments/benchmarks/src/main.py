from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from datasetsforecast.hierarchical import HierarchicalData, HierarchicalInfo
from hierarchicalforecast.core import HierarchicalReconciliation
from hierarchicalforecast.methods import BottomUp, TopDown, MiddleOut, MinTrace, ERM
from statsforecast.core import StatsForecast
from statsforecast.models import Naive, SeasonalNaive, AutoARIMA


def get_reconcilers(group: str, tags_names: List[str]):
    reconcilers = [ 
        BottomUp(),
        MinTrace(method='ols'),
        MinTrace(method='wls_struct'),
        MinTrace(method='wls_var'),
        MinTrace(method='mint_shrink'),
    ]
    return reconcilers

def get_models(seasonality: int):
    models = [
        Naive(),
        SeasonalNaive(season_length=seasonality)
        #AutoARIMA(season_length=seasonality)
    ]
    return models

def main():
    results_dir = Path('./results')
    for group, cls_group in HierarchicalInfo:
        print(f'Dataset: {group}')
        results_group_dir = results_dir / group
        results_group_dir.mkdir(exist_ok=True, parents=True)
        # dataset and problem
        Y_df, S, tags = HierarchicalData.load('./data', group)
        h = cls_group.horizon
        freq = cls_group.freq
        seasonality = cls_group.seasonality
        test_size = cls_group.test_size
        levels = list(range(51, 100, 1))
        # Base forecasts
        base_fcsts_file = results_group_dir / 'base-forecasts.parquet'
        base_fitted_file = results_group_dir / 'base-fitted.parquet'
        if base_fcsts_file.is_file() and base_fitted_file.is_file():
            Y_h = pd.read_parquet(base_fcsts_file)
            Y_fitted = pd.read_parquet(base_fitted_file)
        else:
            models = get_models(seasonality)
            sf = StatsForecast(df=Y_df, models=models, freq=freq, n_jobs=-1)
            Y_h = sf.cross_validation(
                h=h, 
                n_windows=None, 
                test_size=test_size, 
                fitted=True,
                level=levels
            )
            Y_fitted = sf.cross_validation_fitted_values()
            Y_h.to_parquet(base_fcsts_file)
            Y_fitted.to_parquet(base_fitted_file)
        # Reconciliation for each window
        reconcilers = get_reconcilers(group, list(tags.keys()))
        hrec = HierarchicalReconciliation(reconcilers=reconcilers)
        cutoffs = Y_h['cutoff'].unique()
        for intervals_method in ['normality']:
            Y_h_rec = []
            for cutoff in cutoffs:
                Y_h_cutoff = Y_h.query('cutoff == @cutoff').drop(columns=['cutoff'])
                Y_fitted_cutoff = Y_fitted.query('cutoff == @cutoff').drop(columns=['cutoff'])
                Y_h_rec_cutoff = hrec.reconcile(
                    Y_hat_df=Y_h_cutoff, 
                    Y_df=Y_fitted_cutoff, 
                    S=S, 
                    tags=tags,
                    level=levels,
                    intervals_method=intervals_method
                )
                Y_h_rec_cutoff['cutoff'] = cutoff
                Y_h_rec.append(Y_h_rec_cutoff)
            Y_h_rec = pd.concat(Y_h_rec)
            Y_h_rec.to_parquet(results_group_dir / f'reconciled-forecasts-{intervals_method}.parquet')

if __name__=="__main__":
    main()

