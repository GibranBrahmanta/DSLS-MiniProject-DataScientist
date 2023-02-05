'''
- Get external dataset
- Clean jams dataset
    - buang rows yang duplikat
- irregularities dataset 
- merge dataset
REFACTOR: DATA CLEANING DAN FEATURE ENGINEERING JD SATU SENDIRI
'''

import logging
import numpy as np
import pandas as pd
import re

logging.basicConfig(format='%(asctime)s [Create Dataset] %(message)s')
logger = logging.getLogger()
logger.setLevel(logging.INFO)

from helper.data_cleaning import DataCleaner
from helper.feature_engineer import FeatureEngineer
from helper.holiday_data import HolidayScrapper
from helper.weather_data import WeatherScrapper
from utils.config import config

dataset_config = config['dataset']
pipeline_config = config['pipeline']

class DatasetGenerator:

    # Internal Dataset Path
    raw_dataset_path_template = "./data/raw/aggregate_median_{}_Kota {}.csv"
    cleaned_dataset_path_template = "./data/processed/cleaned_aggregate_{}_{}.parquet.gzip"
    temporary_dataset_path_template = "./data/tmp/{}.parquet.gzip"

    # External Dataset Path
    holiday_dataset_path_template = "./data/external/holiday_{}.parquet.gzip"
    weather_dataset_path_template = "./data/external/weather_{}.parquet.gzip"

    def create_dataset(self) -> bool:
        try:
            logger.info("Start process")
            holiday_data = self.open_holiday_dataset()
            for city in dataset_config['city']:
                cleaned_dataset = self.open_cleaned_dataset(city)
                weather_data = self.open_weather_dataset(city, cleaned_dataset['jams'])
                fe = FeatureEngineer(
                    city=city,
                    internal_dataset=cleaned_dataset,
                    weather_data=weather_data,
                    holiday_data=holiday_data
                 )
                fe.create_final_dataset()
            return True
        except Exception as e:
            logger.info("ERROR: {} - {}".format(e.__class__.__name__, str(e)))
            return False
        
    def open_holiday_dataset(self) -> pd.DataFrame:
        if pipeline_config['create_holiday_data']:
            holiday_data = self.create_holiday_dataset()
        else:
            holiday_data = pd.read_parquet(
                self.holiday_dataset_path_template.format(dataset_config['year'])
            )
        return holiday_data
    
    def open_cleaned_dataset(self, city) -> dict:
        if pipeline_config['clean_data'][city]:
            cleaned_dataset = self.clean_dataset(city)
        else:
            cleaned_dataset = {
                'jams': pd.read_parquet(self.cleaned_dataset_path_template.format('jams', city)),
                'irregularities': pd.read_parquet(self.cleaned_dataset_path_template.format('irregularities', city))
            }
        return cleaned_dataset

    def open_weather_dataset(self, city, dataset) -> pd.DataFrame:
        if pipeline_config['get_weather_data'][city]:
            weather_data = self.create_weather_data(city, dataset)
        else:
            weather_data = pd.read_parquet(self.weather_dataset_path_template.format(city))
        return weather_data

    def create_holiday_dataset(self) -> pd.DataFrame:
        logger.info("Start create holiday dataset")
        scrapper = HolidayScrapper(year=dataset_config['year'], path=self.holiday_dataset_path_template)
        result = scrapper.create_dataset()
        if not result:
            raise Exception("Failed on create holiday dataset")
        logger.info("Finish create holiday dataset")
        return result
    
    def clean_dataset(self, city) -> dict:
        logger.info("Start cleaning dataset related to: {}".format(city))
        cleaner = DataCleaner()
        result = cleaner.clean_dataset(city)
        if not result:
            raise Exception("Failed on cleaning dataset")
        logger.info("Finish cleaning dataset")
        return result
    
    def create_weather_data(self, city, dataset) -> pd.DataFrame:
        logger.info("Start create weather dataset related to: {}".format(city))
        coordinate_df = self.compute_coordinate(dataset)
        coordinate_data = {
            'long': np.mean(coordinate_df['mean_long']),
            'lat': np.mean(coordinate_df['mean_lat'])
        }
        scrapper = WeatherScrapper(coordinate_data, city)
        start_timestamp = dataset_config['timestamp']['start_timestamp']
        end_timestamp = dataset_config['timestamp']['end_timestamp']
        result = scrapper.get_weather_data(start_timestamp, end_timestamp)
        if not result:
            raise Exception("Failed on create weather dataset")
        logger.info("Finish create weather dataset related to: {}".format(city))
        return result
    
    def compute_coordinate(self, dataset, city) -> pd.DataFrame:
        data = []
        for _, row in dataset.iterrows():
            data.append(self.get_coordinate_data(row))
        result = pd.DataFrame(data=data, columns=['street', 'mean_long', 'mean_lat'])
        self.save_dataset_to_parquet(result, 
            self.temporary_dataset_path_template.format("{}_coordinate".format(city))
        )
        return result

    def get_coordinate_data(self, row, pattern='(-?(\d+\.?\d+)\s-?(\d+\.?\d+))') -> list:
        street = row['street']
        lst_data = [data[0] for data in re.findall(pattern, row['geometry'])]
        reformatted_data = np.array([np.array(data.split(" ")).astype(float) for data in lst_data])
        mean_long = np.mean(reformatted_data[:,0])
        mean_lat = np.mean(reformatted_data[:,1])
        return [street, mean_long, mean_lat]

    def save_dataset_to_parquet(self, dataset, path) -> bool:
        dataset.to_parquet(path, index=False, compression="gzip")

if __name__ == '__main__':
    gen = DatasetGenerator()
    gen.create_dataset()





