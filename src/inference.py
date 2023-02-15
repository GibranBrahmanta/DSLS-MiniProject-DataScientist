import argparse
import logging
import numpy as np
import pandas as pd

logging.basicConfig(format='%(asctime)s [Inference] %(message)s')
logger = logging.getLogger()
logger.setLevel(logging.INFO)

from data.weather_data import WeatherScrapper
from datetime import datetime, timedelta
from models.classifier import Classifier
from models.time_series_model import Predictor

parser = argparse.ArgumentParser()
parser.add_argument("--city", required=True, help = "City to be forecasted")
parser.add_argument("--year", nargs='*', required=True, help = "Year to be forecasted")
parser.add_argument("--start_timestamp", required=True, help = "Start timestamp")
parser.add_argument("--end_timestamp", required=True, help = "End timestamp")

args = parser.parse_args()

class Inference:

    city_dataset_path_template = "./data/processed/final_dataset_{}.parquet.gzip"
    holiday_dataset_path_template = "./data/external/holiday_{}.parquet.gzip"
    weather_dataset_path_template = "./data/external/weather_{}.parquet.gzip"
    coordinate_dataset_path_template = "./data/interim/{}_coordinate.parquet.gzip"

    time_series_model = {
        'Bogor': 'ARIMA'
    }

    time_series_feature = [
        'median_length', 
        'median_delay', 
        'median_speed_kmh', 
        'median_regular_speed', 
        'median_delay_seconds'
    ]

    classifier_model = {
        'Bogor': 'Bogor_DecisionTree_(criterion_entropy)(splitter_best)(max_depth_10)(max_features_auto)'
    }

    classification_feature = [
        'median_length',	
        'median_delay',	
        'median_speed_kmh',	
        'median_regular_speed',	
        'median_delay_seconds',	
        'rain_intensity',	
        'holiday_gap',
        'level'
    ]

    def __init__(self, city, lst_year) -> None:
        self.city = city
        self.lst_year = lst_year
        self.dataset = self.open_dataset(city)
    
    def open_dataset(self, city) -> dict:
        logger.info("Open related dataset")
        dataset = {
            'traffic': pd.read_parquet(self.city_dataset_path_template.format(city)),
            'holiday': self.open_holiday_dataset(),
            'weather': pd.read_parquet(self.weather_dataset_path_template.format(city)),
            'coordinate': pd.read_parquet(self.coordinate_dataset_path_template.format(city))
        }
        return dataset
    
    def open_holiday_dataset(self) -> pd.DataFrame:
        result = pd.read_parquet(self.holiday_dataset_path_template.format(self.lst_year[0]))
        for i in range(1, len(self.lst_year)):
            tmp = pd.read_parquet(self.holiday_dataset_path_template.format(self.lst_year[i]))
            result = pd.concat([result, tmp], ignore_index=True)
        return result

    def forecast(self, start_timestamp, end_timestamp) -> None:
        start_timestamp = self.convert_to_datetime(start_timestamp)
        end_timestamp = self.convert_to_datetime(end_timestamp)
        dataset = self.create_dataset(start_timestamp, end_timestamp)
        result = self.predict(dataset)
        result.to_csv("huahua.csv", index=False)
        self.save_prediction_result(result, start_timestamp, end_timestamp)
    
    def convert_to_datetime(self, str_time) -> dict:
        str_time = str_time.replace("_", " ") + ".000"
        return datetime.strptime(str_time, '%Y-%m-%d %H:%M:%S.%f')
    
    def create_dataset(self, start_timestamp, end_timestamp) -> pd.DataFrame:
        logger.info("Create used dataset")
        traffic_data = self.dataset['traffic']
        prev_traffic_data = traffic_data[traffic_data['time'] < start_timestamp]
        used_traffic_data = self.predict_traffic_data(prev_traffic_data, start_timestamp, end_timestamp)
        weather_data = self.get_weather_data(start_timestamp, end_timestamp)
        merged_data = used_traffic_data.join(
            weather_data.set_index(['timestamp']),
            on='time',
            how='inner'
        )
        merged_data['holiday_gap'] = merged_data['time'].apply(self.get_nearest_holiday_gap, args=(self.dataset['holiday'],))
        merged_data['level'] = [None] * merged_data.shape[0]
        return merged_data
    
    def get_nearest_holiday_gap(self, time, data) -> int:
        nearest = 365
        for i in range(data.shape[0]):
            curr = np.abs((time - data.iloc[i,1]).days)
            if curr < nearest:
                nearest = np.abs((time - data.iloc[i,1]).days)
        if nearest > 7:
            nearest = -1
        return nearest
    
    def predict_traffic_data(self, prev_traffic_data, start_timestamp, end_timestamp) -> pd.DataFrame:
        logger.info("Create traffic dataset")
        lst_street = list(set(prev_traffic_data['street']))
        latest_timestamp = max(self.dataset['traffic']['time'])
        if start_timestamp > latest_timestamp:
            start_pred = latest_timestamp 
        else:
            start_pred = start_timestamp
        lst_possible_timestamp = self.generate_possible_timestamp(start_pred, end_timestamp, 1)
        used_model = self.time_series_model[self.city]

        data = []
        i = 1
        total = len(lst_street)
        for street in lst_street:
            logger.info("Create traffic dataset on {} ({}/{})".format(street, i, total))
            used_data = prev_traffic_data[prev_traffic_data['street'] == street]
            pred_res = [[street] * len(lst_possible_timestamp), lst_possible_timestamp]
            for feature in self.time_series_feature:
                model = Predictor(
                    city=self.city,
                    model_name=used_model,
                    attr=feature
                )
                res = model.predict(
                    street=street,
                    data=used_data.loc[:, ['time', feature]],
                    timestep=len(lst_possible_timestamp)
                )
                pred_res.append(res)
            row_data = [list(row) for row in zip(*pred_res)]
            data += row_data
            i += 1

        result = pd.DataFrame(data=data, columns=['street', 'time'] + self.time_series_feature)
        result = result[
            (result['time'] >= start_timestamp) & \
                (result['time'] <= end_timestamp)
        ]
        result[self.time_series_feature] = result[self.time_series_feature].clip(0)
        return result
    
    def get_weather_data(self, start_timestamp, end_timestamp):
        logger.info("Get weather data")
        coordinate_data = {
            'long': np.mean(self.dataset['coordinate']['mean_long']),
            'lat': np.mean(self.dataset['coordinate']['mean_lat'])
        }
        weather_data = self.dataset['weather']
        max_stored = max(weather_data['timestamp'])
        lst_possible_timestamp = self.generate_possible_timestamp(start_timestamp, end_timestamp, 1)

        data = []
        latest_timestamp = None
        for timestamp in lst_possible_timestamp:
            logger.info("Get stored weather data on {}".format(datetime.strftime(timestamp, '%Y-%m-%d-%H-%M-%S')))
            latest_timestamp = timestamp
            if timestamp <= max_stored:
                data.append([
                    timestamp,
                    weather_data[weather_data['timestamp'] == timestamp]['rain_intensity'].values[0]
                ])
            else:
                break

        result = pd.DataFrame(data=data, columns=['timestamp', 'rain_intensity'])

        if result.shape[0] != len(lst_possible_timestamp):
            logger.info("Get weather data through API")
            scrapper = WeatherScrapper(coordinate_data, self.city)
            scapper_result = scrapper.get_weather_data(latest_timestamp, end_timestamp, 1, save_dataset=False)
            result = pd.concat([
                result,
                scapper_result.loc[:, ['timestamp', 'rain_intensity']]
            ], ignore_index=True)

        return result
    
    def generate_possible_timestamp(self, start_timestamp, end_timestamp, timestep) -> list:
        result = []
        curr_timestamp = start_timestamp
        while curr_timestamp <= end_timestamp:
            result.append(curr_timestamp)
            curr_timestamp += timedelta(hours=timestep)
        return result

    def predict(self, dataset):
        logger.info("Doing the prediction")
        used_dataset = dataset.loc[:, self.classification_feature]
        used_model = self.classifier_model[self.city]
        classifier = Classifier(
            city=self.city,
            model_name=used_model.split("_")[1],
            model_path=used_model
        )
        pred_res = classifier.predict(used_dataset)
        dataset['level'] = pred_res
        return dataset
    
    def save_prediction_result(self, data, start_timestamp, end_timestamp):
        logger.info("Save the prediction result")
        str_start_timestamp = datetime.strftime(start_timestamp, '%Y-%m-%d-%H-%M-%S')
        str_end_timestamp = datetime.strftime(end_timestamp, '%Y-%m-%d-%H-%M-%S')
        data.to_parquet("./data/inference/{}_{}_{}.parquet.gzip".format(
                self.city, str_start_timestamp, str_end_timestamp
            ),
            index=False,
            compression="gzip"
        )
        
if __name__ == '__main__':
    inf = Inference(city=args.city, lst_year=args.year)
    inf.forecast(
        start_timestamp=args.start_timestamp,
        end_timestamp=args.end_timestamp
    )