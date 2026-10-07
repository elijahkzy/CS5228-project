"""Temporal features used by this model folder. Written by 01_feature_selection.ipynb; rerun that notebook to update it."""

# Stock and economy features selected by L2 stability selection against the trend decomposed from the static features.
# Key = "<training window>/<months the selection used>": "train" = train split only (used while tuning), "train_val" = train + validation
# (used for the final model). The comment is the share of expanding windows in which the feature was among the top 10.
SELECTED_TEMPORAL_FEATURES = {
    'full_history/train': [
        'stk_singapore_post_log_price_mean',  # 0.67
        'stk_sembcorp_log_price_mean',  # 0.88
        'MACRO_SORA_3M_PCT',  # 0.85
        'coe_c_log_quota_premium',  # 0.79
        'stk_sheng_siong_group_log_price_mean',  # 0.82
        'stk_keppel_reit_log_price_mean',  # 0.61
        'stk_jardine_cycle_carriage_log_price_mean',  # 0.64
        'MACRO_PRIVATE_RENT_RPI_YOY',  # 0.73
    ],
    'full_history/train_val': [
        'stk_singapore_airlines_log_price_mean',  # 0.59
        'MACRO_SORA_3M_PCT',  # 0.87
        'stk_sheng_siong_group_log_price_mean',  # 0.85
        'stk_singapore_post_log_price_mean',  # 0.77
        'coe_c_log_quota_premium',  # 0.82
        'stk_jardine_cycle_carriage_log_price_mean',  # 0.64
        'stk_keppel_reit_log_price_mean',  # 0.67
        'stk_genting_singapore_log_price_mean',  # 0.54
        'MACRO_PRIVATE_RENT_RPI_YOY',  # 0.62
    ],
    'from_2023/train': [
        'stk_netlink_trust_log_price_mean',  # 0.80
        'stk_singapore_airlines_log_price_mean',  # 0.50
        'stk_keppel_log_price_mean',  # 1.00
        'stk_singtel_mom_return',  # 1.00
        'stk_genius_group_return_volatility',  # 1.00
        'stk_euda_health_log_price_mean',  # 0.60
        'stk_sembcorp_log_price_mean',  # 0.90
        'stk_comfortdelgro_mom_return',  # 1.00
        'stk_super_hi_international_return_volatility',  # 1.00
        'MACRO_SORA_3M_PCT',  # 1.00
    ],
    'from_2023/train_val': [
        'stk_netlink_trust_log_price_mean',  # 0.50
        'stk_singapore_airlines_log_price_mean',  # 0.69
        'stk_keppel_log_price_mean',  # 1.00
        'stk_sembcorp_log_price_mean',  # 0.62
        'stk_jardine_cycle_carriage_return_volatility',  # 0.56
        'stk_singtel_mom_return',  # 1.00
        'stk_genius_group_return_volatility',  # 1.00
        'MACRO_SORA_3M_PCT',  # 0.75
        'stk_super_hi_international_return_volatility',  # 1.00
        'stk_comfortdelgro_mom_return',  # 1.00
    ],
    'from_2023_06/train': [
        'stk_singtel_log_price_mean',  # 0.60
        'stk_singtel_mom_return',  # 1.00
        'stk_hph_trust_hutchison_port_mom_return',  # 0.80
        'stk_cytomed_therapeutics_return_volatility',  # 1.00
        'stk_sembcorp_mom_return',  # 0.60
        'stk_cmon_return_volatility',  # 0.60
        'stk_jardine_cycle_carriage_return_volatility',  # 0.60
        'stk_hotel_properties_limited_mom_return',  # 0.80
    ],
    'from_2023_06/train_val': [
        'stk_singtel_log_price_mean',  # 0.82
        'stk_hph_trust_hutchison_port_mom_return',  # 0.82
        'stk_karooooo_return_volatility',  # 0.55
        'stk_jardine_cycle_carriage_return_volatility',  # 0.82
        'stk_dbs_group_return_volatility',  # 0.55
        'stk_cytomed_therapeutics_return_volatility',  # 1.00
        'stk_singtel_mom_return',  # 0.64
        'stk_cytomed_therapeutics_mom_return',  # 0.55
        'stk_sembcorp_mom_return',  # 0.73
        'stk_cmon_return_volatility',  # 0.82
    ],
}
