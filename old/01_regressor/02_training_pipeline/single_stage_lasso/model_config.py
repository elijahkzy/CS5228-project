"""Best setup per training window for this model folder, chosen on the validation set (lowest log RMSE) by 02_tune_model.ipynb.
Rerun that notebook to update it."""

BEST_SETUPS = {'full_history/single_stage_lasso': {'lasso_alpha': 0.0001,
                                     'validation': {'log_r2': 0.36496,
                                                    'dollar_r2': 0.42041,
                                                    'log_rmse': 0.18147,
                                                    'dollar_rmse': 488.49287}},
 'from_2023/single_stage_lasso': {'lasso_alpha': 0.001,
                                  'validation': {'log_r2': 0.36682, 'dollar_r2': 0.42156, 'log_rmse': 0.1812, 'dollar_rmse': 488.0098}},
 'from_2023_06/single_stage_lasso': {'lasso_alpha': 0.001,
                                     'validation': {'log_r2': 0.38331,
                                                    'dollar_r2': 0.45732,
                                                    'log_rmse': 0.17883,
                                                    'dollar_rmse': 472.68177}}}
