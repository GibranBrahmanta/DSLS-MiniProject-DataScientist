import logging
import pandas as pd
import sys
import os

sys.path.append(os.getcwd())

logging.basicConfig(format='%(asctime)s [Classification Experiment] %(message)s')
logger = logging.getLogger()
logger.setLevel(logging.INFO)

from classifier import Classifier
from classifier_evaluation import ClassifierEvaluation
from src.utils.config import config

class ClassifierBaselineExperiment:

    dataset_path_template = "./data/processed/final_dataset_{}.parquet.gzip"
    time_series_result_template = "./data/model_result/time_series/{}_{}_{}.parquet.gzip"
    model_result_template = "./data/model_result/classifier/{}_{}.parquet.gzip"
    eval_result_template = "./data/eval_result/classifier/{}_{}.csv"

    lst_city = config['city']
    lst_time_series_model = ['ARIMA']
    lst_time_series_feature = [
        'median_length', 
        'median_delay', 
        'median_speed_kmh', 
        'median_regular_speed', 
        'median_delay_seconds'
    ]
    lst_classification_model = [
        'LogisticRegression',
        'SVM',
        'NaiveBayes',
        'DecisionTree',
        'RandomForest',
        'LightGBM',
        'XGBoost'
    ]
    used_feature = [
        'median_length',	
        'median_delay',	
        'median_speed_kmh',	
        'median_regular_speed',	
        'median_delay_seconds',	
        'rain_intensity',	
        'holiday_gap',
        'level'
    ]

    def run(self) -> None:
        for city in self.lst_city:
            logger.info("Start train classification model on {}".format(city))
            for time_series_model in self.lst_time_series_model:
                logger.info("Using {} model prediction".format(time_series_model))
                dataset = self.open_dataset(city, time_series_model)
                model_dct = self.train(city, dataset)
                model_inference_result = self.inference(city, dataset, model_dct)
                self.evaluate_model(city, model_inference_result)
    
    def open_dataset(self, city, time_series_model) -> dict:
        dataset = {
            'all': pd.read_parquet(self.dataset_path_template.format(city))
        }

        time_series_pred = {}
        for feature in self.lst_time_series_feature:
            time_series_pred[feature] = pd.read_parquet(
                self.time_series_result_template.format(time_series_model, city, feature)
            )

        all_dataset = dataset['all']
        train_dataset = all_dataset[all_dataset['classification_split'] == 'train']
        valid_dataset = all_dataset[all_dataset['classification_split'] == 'valid']
        test_dataset = all_dataset[all_dataset['classification_split'] == 'test']

        valid_dataset = self.change_feature(valid_dataset, time_series_pred)
        test_dataset = self.change_feature(test_dataset, time_series_pred)

        train_dataset = train_dataset.loc[:, self.used_feature]
        valid_dataset = valid_dataset.loc[:, self.used_feature]
        test_dataset = test_dataset.loc[:, self.used_feature]

        dataset['train'] = train_dataset
        dataset['valid'] = valid_dataset
        dataset['test'] = test_dataset

        return dataset
    
    def change_feature(self, dataset, time_series) -> pd.DataFrame:
        result = dataset.copy()
        result.drop(self.lst_time_series_feature, axis=1, inplace=True)
        for feature in self.lst_time_series_feature:
            data = time_series[feature].copy()
            data.drop(['actual'], axis=1, inplace=True)
            data.rename(columns={
                'pred': feature
            }, inplace=True)
            result = result.join(
                data.set_index(['time', 'street']),
                on=['time', 'street'],
                how='inner',
            )
        return result
    
    def train(self, city, dataset) -> dict:
        train_dataset = dataset['train']
        result = {}
        for model_name in self.lst_classification_model:
            logger.info("Start train {} model".format(model_name))
            model = Classifier(
                city=city,
                model_name=model_name
            )
            model.train(train_dataset)
            model.save_model()
            result[model_name] = model
        return result
    
    def inference(self, city, dataset, model_dct) -> dict:
        test_dataset = dataset['test']
        result = {}
        for model_name in self.lst_classification_model:
            logger.info("Start inference using {} model".format(model_name))
            model = model_dct[model_name]
            pred_res = model.predict(test_dataset)
            data = []
            for i in range(test_dataset.shape[0]):
                data.append([
                    test_dataset.iloc[i,-1],
                    pred_res[i]
                ])
            res_df = pd.DataFrame(data=data, columns=['actual', 'pred'])
            res_df.to_parquet(
                self.model_result_template.format(city, model.model_conf),
                index=False,
                compression="gzip"
            )
            result[model.model_conf] = res_df
        return result
    
    def evaluate_model(self, city, model_result) -> None:
        for model_conf in model_result.keys():
            logger.info("Start evaluation on {}".format(model_conf))
            result = model_result[model_conf]
            eval = ClassifierEvaluation(result['actual'], result['pred'])
            eval_res = eval.get_eval_result()
            result_df = pd.DataFrame(data=[[
                model_conf,
                eval_res['accuracy'],
                eval_res['precision'],
                eval_res['recall'],
                eval_res['f1-score']
            ]], columns=['model_conf', 'accuracy', 'precision', 'recall', 'f1-score'])
            result_df.to_csv(
                self.eval_result_template.format(city, model_conf),
                index=False
            )

if __name__ == "__main__":
    exp = ClassifierBaselineExperiment()
    exp.run()