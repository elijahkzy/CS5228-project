label = 'MONTHLY_RENT'

id_col = ['RENT_APPROVAL_YEAR', 'RENT_APPROVAL_MONTH', 'LEASE_COMMENCE_DATE',
       'TOWN', 'BLOCK', 'STREET',
       'POSTAL_CODE', 'LATITUDE', 'LONGITUDE', 'YEAR_COMPLETED',
       'SUBZONE', 'PLANNING_AREA', 'REGION', 'nearest_station', 'nearest_mall']

num_col = ['FLOOR_AREA_SQM','plan_area_ft_total_population',
       'plan_area_ft_pct_age_0_14', 'plan_area_ft_pct_age_15_64',
       'plan_area_ft_pct_age_65_plus', 'plan_area_ft_unemployment_rate',
       'plan_area_ft_total_households', 'plan_area_ft_pct_hdb',
       'plan_area_ft_pct_private_housing', 'plan_area_ft_pct_hh_income_ge_10k',
       'plan_area_ft_pct_hh_income_below_2k',
       'plan_area_ft_avg_household_size', 'plan_area_ft_pct_chinese',
       'plan_area_ft_pct_malay', 'plan_area_ft_pct_indian',
       'plan_area_ft_pct_owner_occupied', 'plan_area_ft_pct_pmet',
       'nearest_station_m', 'n_stations_500m',
       'n_stations_1000m', 'mrt_traffic_tap_in_amount_norm','n_malls_500m', 'nearest_mall_m',
       'mrt_traffic_tap_out_amount_norm', 'mrt_traffic_total_tap_norm',
       'mrt_traffic_weekday_tap_norm', 'mrt_traffic_weekend_tap_norm',
       'mrt_traffic_am_peak_norm', 'mrt_traffic_pm_peak_norm']

cat_col_num = ['MAX_FLOOR',
 'dbscan_cluster',
 'diffusion_cluster',
 'mrt_traffic_busiest_hour',
 'train_lines',
 'station_is_open']

cat_col = ['FLAT_TYPE', 'FLAT_MODEL']

one_hot_col = ['FLAT_TYPE_1 room',
       'FLAT_TYPE_2 room', 'FLAT_TYPE_3 room', 'FLAT_TYPE_4 room',
       'FLAT_TYPE_5 room', 'FLAT_TYPE_executive', 'FLAT_MODEL_2 room',
       'FLAT_MODEL_3gen', 'FLAT_MODEL_adjoined flat', 'FLAT_MODEL_apartment',
       'FLAT_MODEL_dbss', 'FLAT_MODEL_improved',
       'FLAT_MODEL_improved maisonette', 'FLAT_MODEL_maisonette',
       'FLAT_MODEL_model a', 'FLAT_MODEL_model a maisonette',
       'FLAT_MODEL_model a2', 'FLAT_MODEL_new generation',
       'FLAT_MODEL_premium apartment', 'FLAT_MODEL_premium apartment loft',
       'FLAT_MODEL_premium maisonette', 'FLAT_MODEL_simplified',
       'FLAT_MODEL_standard', 'FLAT_MODEL_terrace', 'FLAT_MODEL_type s1',
       'FLAT_MODEL_type s2']