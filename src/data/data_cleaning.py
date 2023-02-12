import logging
import numpy as np
import pandas as pd
import sys
import os

sys.path.append(os.getcwd())

logging.basicConfig(format='%(asctime)s [Cleaning Dataset] %(message)s')
logger = logging.getLogger()
logger.setLevel(logging.INFO)

from src.utils.config import config

dataset_config = config['dataset']

class DataCleaner:

    raw_dataset_path_template = "./data/raw/aggregate_median_{}_Kota {}.csv"
    cleaned_dataset_path_template = "./data/interim/cleaned_aggregate_{}_{}.parquet.gzip"

    def clean_dataset(self, city) -> dict:
        try:
            lst_dataset_type = ['jams', 'irregularities']

            used_street = None
            result = {}
            for dataset_type in lst_dataset_type:
                logger.info("Start cleaning {} dataset".format(dataset_type))
                path = self.raw_dataset_path_template.format(dataset_type, city)
                dataset = pd.read_parquet(path)
                dataset_non_duplicated = self.remove_duplicate(dataset)
                if dataset_type == 'jams':
                    used_dataset, used_street = self.filter_street(dataset_non_duplicated, 
                        threshold=dataset_config['preprocessing']['completion_threshold']
                    )
                else:
                    used_dataset, used_street = self.filter_street(dataset_non_duplicated, 
                        threshold=dataset_config['preprocessing']['completion_threshold'],
                        used_street=used_street
                    )
                self.save_dataset_to_parquet(used_dataset,
                    self.cleaned_dataset_path_template.format(dataset_type, city)
                )
                result['dataset_type'] = used_dataset
                logger.info("Finish cleaning {} dataset".format(dataset_type))
            
            return result
        except Exception as e:
            logger.info("ERROR: {} - {}".format(e.__class__.__name__, str(e)))
            return None
    
    def remove_duplicate(self, dataset) -> pd.DataFrame:
        count_group_jam = pd.DataFrame({'count':dataset.groupby(by=['street', 'time']).size()})

        result = pd.DataFrame(columns=dataset.columns)
        counter = 0
        for time, street in zip(count_group_jam['time'], count_group_jam['street']):
            progress = (counter/count_group_jam.shape[0]) * 100
            if progress % 10 == 0:
                logger.info("Current progress: {}%".format(progress))
            data = dataset[(dataset['time'] == time) & (dataset['street'] == street)]
            if data.shape[0] != 1:
                data.sort_values(by=['total_records'], ascending=False, inplace=True)
            result = result.append(data.iloc[0,:], ignore_index=True)
            counter += 1
    
        return result
    
    def filter_street(self, dataset, threshold, used_street=None) -> tuple[pd.DataFrame, list]:
        if not used_street:
            completion_rate_data = self.compute_completion_rate(dataset)
            used_street = self.get_used_street(completion_rate_data, threshold)
        return dataset[dataset['street'].isin(used_street)], used_street
    
    def compute_completion_rate(self, dataset) -> pd.DataFrame:
        street_count = dataset['street'].value_counts().to_dict()
        day = (max(dataset['time']) - min(dataset['time'])).days
        hour = 24
        max_row = day * hour

        data = []
        for street in street_count.keys():
            data.append([
                street,
                street_count[street],
                street_count[street]/max_row
            ])
        result = pd.DataFrame(data=data, columns=['street', 'count', 'completion_rate'])
        
        return result
    
    def get_used_street(self, completion_rate_data, threshold) -> list:
        thres = np.quantile(completion_rate_data['completion_rate'], q=threshold)
        used_street = list(completion_rate_data[completion_rate_data['completion_rate'] >= thres]['street'])
        return used_street
    
    def save_dataset_to_parquet(self, dataset, path) -> bool:
        dataset.to_parquet(path, index=False, compression="gzip")