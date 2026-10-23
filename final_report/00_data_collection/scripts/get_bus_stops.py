import requests
import requests
import pandas as pd
import json
import requests
import socket
import pandas as pd
from pandarallel import pandarallel
import os 
import pathlib

results = requests.get(f"""http://datamall2.mytransport.sg/ltaodataservice/BusStops""")

if __name__=="__main__":
    #Authentication parameters
    headers = { 'AccountKey' : 'YOUR ACCOUNT KEY HERE',
                'accept' : 'application/json'} 

    #API parameters
    method = 'GET'
    body = ''

    all_bus_stops = []
    #Obtain results
    for i in range(0,100000,500):
        results = requests.get(
                            f'''http://datamall2.mytransport.sg/ltaodataservice/BusStops?$skip={i}''',
                            headers = headers)\
                            .json()['value']
        if len(results) == 0:
            break
        else:
            all_bus_stops += results
    #Parse JSON to print
    df = pd.DataFrame(all_bus_stops)
    df.to_csv('sg-bus-stops.csv')