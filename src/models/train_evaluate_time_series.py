import logging
import pandas as pd
import random
import warnings

logging.basicConfig(format='%(asctime)s [Time Series Experiment] %(message)s')
logger = logging.getLogger()
logger.setLevel(logging.INFO)

warnings.filterwarnings('ignore')

from time_series_model import Predictor
from time_series_evaluation import Evaluation
from utils.config import config

class TimeSeriesExperiment:

    dataset_path_template = "./data/processed/final_dataset_{}.parquet.gzip"
    model_result_template = "./data/model_result/time_series/{}_{}_{}.parquet.gzip"
    eval_result_template = "./data/eval_result/time_series/{}_{}_{}_{}.csv"

    lst_model = ['ARIMA']
    lst_feature = [
        'median_length', 
        'median_delay', 
        'median_speed_kmh', 
        'median_regular_speed', 
        'median_delay_seconds'
    ]
    lst_city = config['city']
    
    def open_dataset(self, city):
        dataset = pd.read_parquet(self.dataset_path_template.format(city))
        return dataset
    
    def pipeline(self):
        for city in self.lst_city:
            dataset = self.open_dataset(city)
            self.modeling(dataset)
            self.inference(dataset)
            self.evaluate_model()
    
    def modeling(self, dataset):
        train_dataset = dataset[dataset['time_series_split'] == 'train']

        for model_name in self.lst_model:
            for feature in self.lst_feature:
                logger.info("Start train {} model on {}".format(model_name, feature))
                model = Predictor(
                    city=self.city,
                    model_name=model_name,
                    attr=feature
                )
                model.train(train_dataset)
    
    def inference(self, dataset):
        train_dataset = dataset[dataset['time_series_split'] == 'train']
        test_dataset = dataset[dataset['time_series_split'] == 'test']

        lst_street = list(set(train_dataset['street']))

        for model_name in self.lst_model:
            for feature in self.lst_feature:
                logger.info("Start inference {} model on {}".format(model_name, feature))
                model = Predictor(
                    city=self.city,
                    model_name=model_name,
                    attr=feature
                )
                data = []
                for street in lst_street:
                    used_train_data = train_dataset[train_dataset['street'] == street]
                    used_test_data = test_dataset[test_dataset['street'] == street]

                    timestep = used_test_data.shape[0]
                    pred_res = model.predict(
                        street=street,
                        data=used_train_data.loc[:,['time', feature]],
                        timestep=timestep
                    )
                    for i in range(timestep):
                        data.append([
                            street,
                            used_test_data.iloc[i,:]['time'],
                            used_test_data.iloc[i,:][feature],
                            pred_res[i]
                            
                        ])
                result_df = pd.DataFrame(data=data, columns=['street', 'time', 'actual', 'pred'])
                result_df.to_parquet(
                    self.model_result_template.format(model_name, self.city, feature),
                    index=False,
                    compression="gzip"
                )

    def evaluate_model(self):
        for model_name in self.lst_model:
            for feature in self.lst_feature:
                logger.info("Start evaluation {} model on {}".format(model_name, feature))
                model_result = pd.read_parquet(self.model_result_template.format(model_name, self.city, feature))
                overall_eval = Evaluation(model_result['actual'], model_result['pred'])
                overall_res = overall_eval.get_eval_result()
                overall_df = pd.DataFrame(
                    data=[[
                        model_name,
                        overall_res['rmse'],
                        overall_res['mae']
                    ]],
                    columns=['model_name', 'rmse', 'mae']
                )
                overall_df.to_csv(
                    self.eval_result_template.format(model_name, self.city, feature, 'overall'),
                    index=False
                )
                self.evaluate_per_category(model_result, model_name, feature, 'time')
                self.evaluate_per_category(model_result, model_name, feature, 'street')
    
    def evaluate_per_category(self, model_result, model_name, feature, category):
        lst_category = list(set(model_result[category]))
        data = []
        for cat in lst_category:
            used_result = model_result[model_result[category] == cat]
            eval = Evaluation(used_result['actual'], used_result['pred'])
            eval_res = eval.get_eval_result()
            data.append([
                cat,
                eval_res['rmse'],
                eval_res['mae']
            ])
        result_df = pd.DataFrame(
            data=data,
            columns=[category, 'rmse', 'mae']
        )
        result_df.to_csv(
            self.eval_result_template.format(model_name, self.city, feature, category),
            index=False
        )

if __name__ == '__main__':
    exp = TimeSeriesExperiment()
    exp.pipeline()