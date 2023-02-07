import logging
import numpy as np
import pandas as pd
import sys
import os

sys.path.append(os.getcwd())

logging.basicConfig(format='%(asctime)s [Feature Engineering] %(message)s')
logger = logging.getLogger()
logger.setLevel(logging.INFO)

from datetime import datetime, timedelta
from random import randint
from src.utils.config import config

class FeatureEngineer:

    dataset_config = config['dataset']
    modeling_config = config['modeling']
    pipeline_config = config['pipeline']
    lst_city = config['city']

    cleaned_dataset_path_template = "./data/interim/cleaned_aggregate_{}_{}.parquet.gzip"
    complete_dataset_template = "./data/interim/complete_aggregate_{}_{}.parquet.gzip"
    final_dataset_template = "./data/processed/final_dataset_{}.parquet.gzip"

    holiday_dataset_path_template = "./data/external/holiday_{}.parquet.gzip"
    weather_dataset_path_template = "./data/external/weather_{}.parquet.gzip"

    speed_constanta = {
        1: [61, 80],
        2: [41, 60],
        3: [21, 40],
        4: [1, 20]
    }

    start_timestamp = datetime.strptime(dataset_config['timestamp']['start_timestamp'], '%Y-%m-%d %H:%M:%S.%f')
    end_timestamp = datetime.strptime(dataset_config['timestamp']['end_timestamp'], '%Y-%m-%d %H:%M:%S.%f')

    time_series_split = {
        'train': datetime.strptime(modeling_config['time_series']['train_set'], '%Y-%m-%d %H:%M:%S.%f'),
        'test': datetime.strptime(modeling_config['time_series']['test_set'], '%Y-%m-%d %H:%M:%S.%f')
    }

    classification_split = {
        'train': datetime.strptime(modeling_config['classification']['train_set'], '%Y-%m-%d %H:%M:%S.%f'),
        'valid': datetime.strptime(modeling_config['classification']['valid_set'], '%Y-%m-%d %H:%M:%S.%f'),
        'test': datetime.strptime(modeling_config['classification']['test_set'], '%Y-%m-%d %H:%M:%S.%f')
    }   
    
    def build_feature(self) -> None:
        try:
            for city in self.lst_city:
                if self.pipeline_config['build_feature'][city]:
                    logger.info("Start creating final dataset on {}".format(city))
                    completed_jam = self.process_jam_dataset(city)
                    completed_irregularities = self.process_irregularities_dataset(city)
                    self.create_final_dataset(completed_jam, completed_irregularities, city)
                    logger.info("Finish creating final dataset on {}".format(city))
        except Exception as e:
            logger.info("ERROR: {} - {}".format(e.__class__.__name__, str(e)))
            return None
    
    def process_jam_dataset(self, city) -> None:
        logger.info("Start completing jams dataset")
        used_col = [
            'time',
            'street',
            'level',
            'median_length',
            'median_delay',
            'median_speed_kmh'
        ]
        df_jam = pd.read_parquet(self.cleaned_dataset_path_template('jams', city))
        df_jam = df_jam.loc[:, used_col]

        lst_street = list(set(df_jam['street']))

        completed_jam = pd.DataFrame(columns=df_jam.columns)

        curr = 1
        total = len(lst_street)
        for street in lst_street:
            logger.info("Completing jams dataset related to {} ({}/{})".format(street, curr, total))
            used_data = df_jam[df_jam['street'] == street]
            used_data.sort_values(by=['time'], 
                ascending=True, 
                inplace=True
            )

            street_data = pd.DataFrame(columns=df_jam.columns)
            current_timestamp = self.start_timestamp
            
            while current_timestamp <= self.end_timestamp:
                current_data = used_data[used_data['time'] == current_timestamp]
                related_data, is_external = self.get_related_data(used_data, current_timestamp, df_jam)
                if current_data.shape[0] == 0:
                    median_speed = self.get_median_speed(related_data)
                    median_length = self.get_median_length(related_data, median_speed)
                    median_delay = self.get_median_delay(related_data, median_speed)
                    current_data = pd.DataFrame(data=[[
                        current_timestamp,
                        street,
                        0,
                        median_length,
                        median_delay,
                        median_speed
                    ]], columns=df_jam.columns)
                elif current_data['median_delay'].values[0] == -1:
                    current_data['median_delay'] = self.clean_delay_data(related_data, is_external)
                street_data = street_data.append(current_data, ignore_index=True)
                current_timestamp += timedelta(seconds=3600)

            completed_jam = completed_jam.append(
                street_data,
                ignore_index=True
            )
            curr += 1

        self.jam_dataset = completed_jam
        self.save_dataset_to_parquet(
            completed_jam,
            self.complete_dataset_template("jams", self.city)
        )
        logger.info("Finish completing jams dataset")

    def clean_delay_data(self, data, is_external) -> float:
        min_level = min(data['level'])
        if min_level == 5 or is_external:
            return 850.0 + randint(100, 500)
        else:
            used_data = data[data['level'] == min_level]
            min_data = used_data[used_data['median_speed_kmh'] == min(used_data['median_speed_kmh'])].iloc[0,:]
            gap = min_data['median_speed_kmh'] - 0.0
            delay_close_to_0 = min_data['median_delay']
            constanta = np.mean(
                [row['median_delay']/row['median_speed_kmh'] for _, row in used_data.iterrows()]
            )
            delay_to_0 = constanta * gap
            return delay_close_to_0 + delay_to_0 + randint(1, 10)

    def get_related_data(self, data, timestamp, all_data) -> tuple[pd.DataFrame, bool]:
        result = data[data['time'] < timestamp]
        is_external = False
        if result.shape[0] == 0 or min(result['level']) == 5:
            result = data[data['time'] > timestamp].iloc[0:5]
        if min(result['level']) == 5:
            result = all_data[all_data['time'] < timestamp]
            is_external = True
        return result, is_external

    def get_median_speed(self, data) -> float:
        min_level = min(data['level'])
        if min_level < 5:
            used_speed = np.mean(data[data['level'] == min_level]['median_speed_kmh'])
            constanta = randint(self.speed_constanta[min_level][0], self.speed_constanta[min_level][1])/100
            result = used_speed / constanta
        else:
            result = 15.32 # https://publikasiilmiah.ums.ac.id/bitstream/handle/11617/8159/B79_Robby%20Hartono.pdf
        return result

    def get_median_length(self, data, current_median_speed) -> float:
        min_level = min(data['level'])
        used_data = data[data['level'] == min_level]
        constanta = np.mean(
            [row['median_length']/row['median_speed_kmh'] for _, row in used_data.iterrows()]
        )
        return constanta * current_median_speed

    def get_median_delay(self, data, current_median_speed) -> float:
        min_level = min(data['level'])
        used_data = data[data['level'] == min_level]
        constanta = np.mean(
            [row['median_delay']/row['median_speed_kmh'] for _, row in used_data.iterrows()]
        )
        return (constanta/(10*min_level)) * current_median_speed
    
    def process_irregularities_dataset(self, city) -> None:
        logger.info("Start completing irregularities dataset")
        used_col = [
            'time',
            'street',
            'median_regular_speed',
            'median_delay_seconds'
        ]
        df_irregularities = pd.read_parquet(self.cleaned_dataset_path_template.format('irregularities', city))
        df_irregularities = df_irregularities.loc[:, used_col]

        lst_street = list(set(self.jam_dataset['street']))

        completed_irregularities = pd.DataFrame(columns=df_irregularities.columns)

        curr = 1
        total = len(lst_street)
        for street in lst_street:
            logger.info("Completing jam dataset related to {} ({}/{})".format(street, curr, total))
            street_df = df_irregularities[df_irregularities['street'] == street]

            street_data = pd.DataFrame(columns=df_irregularities.columns)
            current_timestamp = self.start_timestamp

            while current_timestamp <= self.end_timestamp:
                irregularities_data = street_df[street_df['time'] == current_timestamp]
                if irregularities_data.shape[0] == 0:
                    jam_data = self.jam_dataset[
                        (self.jam_dataset['street'] == street) & \
                            (self.jam_dataset['time'] <= current_timestamp)
                    ]
                    median_regular_speed = self.get_median_data(jam_data, 'median_speed_kmh')
                    median_delay_seconds = self.get_median_data(jam_data, 'median_delay')
                    irregularities_data = pd.DataFrame(data=[[
                        current_timestamp,
                        street,
                        median_regular_speed,
                        median_delay_seconds
                    ]], columns=df_irregularities.columns)
                street_data = street_data.append(irregularities_data, ignore_index=True)
                current_timestamp += timedelta(seconds=3600)
    
            completed_irregularities = completed_irregularities.append(street_data, ignore_index=True)
            curr += 1
        
        self.irregularities_dataset = completed_irregularities
        self.save_dataset_to_parquet(
            completed_irregularities,
            self.complete_dataset_template("irregularities", self.city)
        )
        logger.info("Finish completing irregularities dataset")

    def get_median_data(self, data, column) -> float:
        if data.shape[0] == 1:
            return data[column].values[0]
        return np.quantile(data.iloc[:-1,:][column], q=0.5)
    
    def create_final_dataset(self, jam_dataset, irregularities_dataset, city) -> None:
        logger.info("Merging all dataset")
        holiday_data = pd.read_parquet(self.holiday_dataset_path_template.format(self.dataset_config['year']))
        weather_data = pd.read_parquet(self.weather_dataset_path_template.format(city))
        final_dataset = jam_dataset.join(
            irregularities_dataset.set_index(['time', 'street']),
            on=['time', 'street'],
            how='inner'
        )
        final_dataset = final_dataset.join(
            weather_data.set_index(["timestamp"]),
            on=['time'],
            how='inner'
        )
        final_dataset['holiday_gap'] = final_dataset['time'].apply(self.get_nearest_holiday_gap, args=(holiday_data,))
        final_dataset.reset_index(inplace=True, drop=True)
        logger.info("Create Additional Features")
        final_dataset['time_series_split'] = final_dataset['time'].apply(self.get_time_series_split, args=(self.time_series_split,))
        final_dataset['classification_split'] = final_dataset['time'].apply(self.get_classification_split, args=(self.classification_split,))
        self.save_dataset_to_parquet(
            final_dataset,
            self.final_dataset_template.format(city)
        )

    def get_nearest_holiday_gap(self, time, data) -> int:
        nearest = 365
        for i in range(data.shape[0]):
            curr = np.abs((time - data.iloc[i,1]).days)
            if curr < nearest:
                nearest = np.abs((time - data.iloc[i,1]).days)
        if nearest > 7:
            nearest = -1
        return nearest

    def get_time_series_split(self, time, data) -> str:
        if time >= data['train'] and time < data['test']:
            return 'train'
        else:
            return 'test'

    def get_classification_split(self, time, data) -> str:
        if time >= data['train'] and time < data['valid']:
            return 'train'
        elif time >= data['valid'] and time < data['test']:
            return 'valid'
        else:
            return 'test'

    def save_dataset_to_parquet(self, dataset, path) -> bool:
        dataset.to_parquet(path, index=False, compression="gzip")  



    
