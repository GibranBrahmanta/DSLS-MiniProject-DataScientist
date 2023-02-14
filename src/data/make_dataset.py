import logging
import numpy as np
import pandas as pd
import re

logging.basicConfig(format='%(asctime)s [Create Dataset] %(message)s')
logger = logging.getLogger()
logger.setLevel(logging.INFO)

from data_cleaning import DataCleaner
from datetime import datetime
from holiday_data import HolidayScrapper
from utils.config import config
from weather_data import WeatherScrapper

class DatasetGenerator:

    cleaned_dataset_path_template = "./data/interim/cleaned_aggregate_{}_{}.parquet.gzip"
    temporary_dataset_path_template = "./data/interim/{}.parquet.gzip"

    dataset_config = config['dataset']
    pipeline_config = config['pipeline']
    lst_city = config['city']

    start_timestamp = datetime.strptime(dataset_config['timestamp']['start_timestamp'], '%Y-%m-%d %H:%M:%S.%f')
    end_timestamp = datetime.strptime(dataset_config['timestamp']['end_timestamp'], '%Y-%m-%d %H:%M:%S.%f')

    def create_dataset(self) -> bool:
        try:
            logger.info("Start process")
            self.get_holiday_dataset()
            for city in self.lst_city:
                cleaned_dataset = self.open_cleaned_dataset(city)
                self.get_weather_dataset(city, cleaned_dataset['jams'])
            return True
        except Exception as e:
            logger.info("ERROR: {} - {}".format(e.__class__.__name__, str(e)))
            return False
        
    def get_holiday_dataset(self) -> None:
        if self.pipeline_config['create_holiday_data']:
            self.create_holiday_dataset()
    
    def open_cleaned_dataset(self, city) -> dict:
        if self.lst_citypipeline_config['clean_data'][city]:
            cleaned_dataset = self.clean_dataset(city)
        else:
            cleaned_dataset = {
                'jams': pd.read_parquet(self.cleaned_dataset_path_template.format('jams', city)),
                'irregularities': pd.read_parquet(self.cleaned_dataset_path_template.format('irregularities', city))
            }
        return cleaned_dataset

    def get_weather_dataset(self, city, dataset) -> None:
        if self.pipeline_config['get_weather_data'][city]:
            self.create_weather_data(city, dataset)

    def create_holiday_dataset(self) -> None:
        logger.info("Start create holiday dataset")
        scrapper = HolidayScrapper(year=self.dataset_config['year'], path=self.holiday_dataset_path_template)
        result = scrapper.create_dataset()
        if not result:
            raise Exception("Failed on create holiday dataset")
        logger.info("Finish create holiday dataset")
    
    def clean_dataset(self, city) -> dict:
        logger.info("Start cleaning dataset related to: {}".format(city))
        cleaner = DataCleaner()
        result = cleaner.clean_dataset(city)
        if not result:
            raise Exception("Failed on cleaning dataset")
        logger.info("Finish cleaning dataset")
        return result
    
    def create_weather_data(self, city, dataset) -> None:
        logger.info("Start create weather dataset related to: {}".format(city))
        coordinate_df = self.compute_coordinate(dataset)
        coordinate_data = {
            'long': np.mean(coordinate_df['mean_long']),
            'lat': np.mean(coordinate_df['mean_lat'])
        }
        scrapper = WeatherScrapper(coordinate_data, city)
        result = scrapper.get_weather_data(self.start_timestamp, self.end_timestamp, 1)
        if not result:
            raise Exception("Failed on create weather dataset")
        logger.info("Finish create weather dataset related to: {}".format(city))
    
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





