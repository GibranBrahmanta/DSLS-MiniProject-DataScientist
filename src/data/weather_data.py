import calendar
import logging
import pandas as pd
import requests
import sys
import os

sys.path.append(os.getcwd())

logging.basicConfig(format='%(asctime)s [Get Weather Data] %(message)s')
logger = logging.getLogger()
logger.setLevel(logging.INFO)

from datetime import datetime, timedelta
from src.utils.config import config

dataset = config['dataset']

class WeatherScrapper:

    api_key = dataset['weather']['api_key']
    gmt = dataset['weather']['gmt']
    url_template = "http://api.openweathermap.org/data/3.0/onecall/timemachine?lat={}&lon={}&dt={}&appid={}"

    def __init__(self, location_data, city) -> None:
        self.location_data = location_data
        self.city = city
    
    def get_weather_data(self, start_timestamp, end_timestamp, timestep, save_dataset=True) -> pd.DataFrame:
        logger.info("Start process")
        try:
            lst_timestamp = self.generate_possible_timestamp(start_timestamp, end_timestamp, timestep)
            lst_weather_condition = self.get_weather_condition(lst_timestamp)
            dataset = self.save_dataset(lst_weather_condition, start_timestamp, end_timestamp, save_dataset)
            logger.info("Finish process")
            return dataset
        except Exception as e:
            logger.info("ERROR: {} - {}".format(e.__class__.__name__, str(e)))
            return None
    
    def generate_possible_timestamp(self, start_timestamp, end_timestamp, timestep) -> list:
        result = []
        curr_timestamp = start_timestamp
        while curr_timestamp <= end_timestamp:
            result.append(curr_timestamp)
            curr_timestamp += timedelta(hours=timestep)
        logger.info("Finish generate possible timestamp")
        return result
    
    def convert_timestamp_to_unix(self, timestamp) -> float:
        utc_timestamp = timestamp - timedelta(hours=self.gmt) 
        result = calendar.timegm(utc_timestamp.timetuple())
        return result
    
    def get_weather_condition(self, lst_timestamp) -> list:
        result = []
        for timestamp in lst_timestamp:
            unix_timestamp = self.convert_timestamp_to_unix(timestamp)
            response = self.get_response(unix_timestamp)['data'][0]
            if 'rain' in response.keys():
                rain = response['rain']['1h']
            else:
                rain = 0.0
            result.append([
                timestamp,
                response['temp'],
                response['feels_like'],
                response['pressure'],
                response['humidity'],
                response['dew_point'],
                response['clouds'],
                response['wind_speed'],
                response['wind_deg'],
                response['weather'][0]['main'],
                rain
            ])
            logger.info("Finish get weather data on {}".format(timestamp))
        return result 

    def get_response(self, timestamp) -> dict:
        url = self.url_template.format(
            self.location_data['lat'],
            self.location_data['long'],
            timestamp,
            self.api_key
        )
        return requests.get(url).json()
    
    def save_dataset(self, lst_weather_condition, start_timestamp, end_timestamp, save_dataset) -> pd.DataFrame:
        df = pd.DataFrame(data=lst_weather_condition, columns=[
            'timestamp',
            'temp',
            'feels_like',
            'pressure',
            'humidity',
            'dew_point',
            'clouds',
            'wind_speed',
            'wind_deg',
            'weather_type',
            'rain_intensity'
        ])
        str_start_timestamp = datetime.strftime(start_timestamp, '%Y-%m-%d-%H-%M-%S')
        str_end_timestamp = datetime.strftime(end_timestamp, '%Y-%m-%d-%H-%M-%S')
        if save_dataset:
            df.to_parquet("./data/external/weather_{}_{}_{}.parquet.gzip".format(
                    self.city, str_start_timestamp, str_end_timestamp
                ),
                index=False,
                compression="gzip"
            )
        logger.info("Finish save data")
        return df