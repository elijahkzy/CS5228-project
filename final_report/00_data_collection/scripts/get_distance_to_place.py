import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))      # final_report: data/config.py holds every path of the project
from data import config as cfg      # noqa: E402
import requests
import geopy.distance
import pandas as pd
from tqdm import tqdm
from pandarallel import pandarallel
from sys import argv

tqdm.pandas()
token1 = 'YOUR API KEY HERE'

def get_travel_time(row, end_coords):
    start_coords = str(row['approx_lat']) + ',' + str(row['approx_lon'])

    params = {
        "date": "09-09-2026",
        "mode": "TRANSIT",
        "maxWalkDistance": 1000,
        "numItineraries": 3,
        "routeType": "pt",
        "time": "08:00:00",
        "start": start_coords,
        "end": end_coords,
        }

    headers = {"Authorization": token1}

    response = requests.get(
        "https://www.onemap.gov.sg/api/public/routingsvc/route",
        params=params,
        headers=headers,
        timeout=30,
    )
    
    if response.status_code != 200:
        print(response.status_code)
        return {'street' : row['street'],
                'start_coords' : start_coords,
                'end_coords' : end_coords,
                'time_taken_by_public_transport' : 0,
                'transfers' : 0,
                'walking_distance' : 0,
                'straight_line_dist' : 0}
    else:
        d = response.json()
        start_coords_num = (start_coords.split(',')[0],start_coords.split(',')[1])
        end_coords_num = (end_coords.split(',')[0],end_coords.split(',')[1])
        walking_distance = d['plan']['itineraries'][0]['walkDistance']
        time_taken_by_public_transport = d['plan']['itineraries'][0]['duration']
        no_of_transfers = d['plan']['itineraries'][0]['transfers']
        return {'street' : row['street']
                ,'start_coords' : start_coords,
                'end_coords' : end_coords,
                'time_taken_by_public_transport' : time_taken_by_public_transport,
                'transfers' : no_of_transfers,
                'walking_distance' : walking_distance,
                'straight_line_dist' : geopy.distance.geodesic(start_coords_num, end_coords_num).m}

def get_travel_time_df(query_df, end_coords, parallel):
    df = query_df.copy()
    
    if parallel:
        pandarallel.initialize()
        df_travel_time = df.parallel_apply(get_travel_time, end_coords =end_coords, axis = 1)\
                        .apply(pd.Series)
    else:
        df_travel_time = df.progress_apply(get_travel_time, end_coords =end_coords, axis = 1)\
                        .apply(pd.Series)
    
    return(df_travel_time)

if __name__ == '__main__':
    coords_dict = { 'cbd' : '1.3051639,103.8499352',
                    'nus' : '1.2933881,103.7820812',
                    'stadium' : '1.30284,103.8731613',
                    'toapayoh' : '1.3364408,103.8536688'}
    
    location = coords_dict[argv[1]]
    compute_in_parallel = argv[2] == 'y'

    hdb_df = pd.read_csv(cfg.HDB_TRAVEL_TIME_INPUT_CSV)[['street','approx_lat','approx_lon']]
    hdb_df_dist = get_travel_time_df(hdb_df, location, compute_in_parallel)
    hdb_df_dist.to_csv(cfg.PROCESSED_DIR / f"hdb_df_{argv[1]}_dist.csv",index=False)
