"""Best setup per training window for this model folder, chosen on the validation set (lowest log RMSE) by 02_tune_model.ipynb.
Rerun that notebook to update it."""

BEST_SETUPS = {'full_history/single_stage_ridge': {'ridge_alpha': 100.0,
                                     'validation': {'log_r2': 0.3599, 'dollar_r2': 0.41701, 'log_rmse': 0.18219, 'dollar_rmse': 489.92469}},
 'from_2023/single_stage_ridge': {'ridge_alpha': 10000.0,
                                  'validation': {'log_r2': 0.36909, 'dollar_r2': 0.42569, 'log_rmse': 0.18088, 'dollar_rmse': 486.26267}},
 'from_2023_06/single_stage_ridge': {'ridge_alpha': 5000.0,
                                     'validation': {'log_r2': 0.37897,
                                                    'dollar_r2': 0.45219,
                                                    'log_rmse': 0.17946,
                                                    'dollar_rmse': 474.91227}}}
