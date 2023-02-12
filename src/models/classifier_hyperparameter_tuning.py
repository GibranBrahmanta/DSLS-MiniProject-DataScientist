import itertools
import logging
import pandas as pd
import sys
import os

sys.path.append(os.getcwd())

logging.basicConfig(format='%(asctime)s [Classification Hyperparameter Tuning Experiment] %(message)s')
logger = logging.getLogger()
logger.setLevel(logging.INFO)

from classifier import Classifier
from classifier_evaluation import ClassifierEvaluation
from src.utils.config import config

modeling_config = config['modeling']['classification']

class ClassifierHyperparameterExperiment:

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
    lst_classification_model = [
        'RandomForest',
        'LightGBM',
        'XGBoost'
    ]
    param_dct = {
        'LogisticRegression': {
            'penalty': ['l1', 'l2', 'elasticnet'],
            'C': [1.0, 0.1, 0.01],
            'solver': ['newton-cg', 'sag', 'saga', 'lbfgs']
        },
        'SVM': {
            'C': [1.0, 0.1, 0.01],
            'multi_class': ['ovr', 'crammer_singer'],
            'loss': ['hinge', 'squared_hinge']
        },
        'NaiveBayes': {
            'alpha': [1.0, 0.1, 0.01],
            'norm': [True, False]
        },
        'DecisionTree': {
            'criterion': ['gini', 'entropy', 'log_loss'],
            'splitter': ['best', 'random'],
            'max_depth': [10, 25, 50, 75, 100],
            'max_features': ['auto', 'sqrt', 'log2'] 
        },
        'RandomForest': {
            'criterion': ['gini', 'entropy', 'log_loss'],
            'n_estimators': [100, 250, 500],
            'max_depth': [10, 50, 100],
            'max_features': ['auto', 'sqrt', 'log2'] 
        },
        'LightGBM': {
            'learning_rate': [0.1, 0.01, 0.001],
            'n_estimators':  [100, 250, 500],
            'max_depth': [10, 50, 100],
            'objective': ['multiclass']
        },
        'XGBoost': {
            'n_estimators':  [100, 250, 500],
            'max_depth': [10, 50, 100],
            'learning_rate': [0.1, 0.01, 0.001]
        }
    }
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
    top_model = modeling_config['top_n']

    def run(self) -> None:
        for city in self.lst_city:
            logger.info("Start train classification model on {}".format(city))
            for time_series_model in self.lst_time_series_model:
                logger.info("Using {} model prediction".format(time_series_model))
                dataset = self.open_dataset(city, time_series_model)
                for model_name in self.lst_classification_model:
                    logger.info("Start hyperparameter tuning on {}".format(model_name))
                    self.model_tuning_pipeline(city, model_name, dataset)
    
    def open_dataset(self, city, time_series_model) -> dict:
        logger.info("Start open dataset related to {} using {} result".format(city, time_series_model))
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
    
    def get_parameter_dct(self, param_dct):
        lst_param = [param_dct[key] for key in param_dct.keys()]
        lst_comb = list(itertools.product(*lst_param))
        result = []
        lst_key = list(param_dct.keys())
        for comb in lst_comb:
            dct_comb = {lst_key[i]:comb[i] for i in range(len(lst_key))}
            result.append(dct_comb)
        return result
    
    def model_tuning_pipeline(self, city, model_name, dataset):
        lst_param = self.get_parameter_dct(self.param_dct[model_name])
        trained_model = self.train_model(city, model_name, lst_param, dataset['train'])
        valid_result = self.evaluate_model(city, model_name, trained_model, dataset['valid'], 'valid')
        filtered_model = self.save_trained_model(valid_result, trained_model)
        self.evaluate_model(city, model_name, filtered_model, dataset['test'], 'test')
    
    def train_model(self, city, model_name, lst_param, data):
        logger.info("Start training")
        result = {}
        n = 1
        total = len(lst_param)
        for param in lst_param:
            logger.info("Train progress: {}/{}".format(n, total))
            model = Classifier(
                city=city,
                model_name=model_name,
                param=param
            )
            try:
                model.train(data)
                result[model.model_conf] = model
            except:
                logger.info("Invalid parameter combination: {}".format(model.model_conf))
            n += 1
        return result
    
    def evaluate_model(self, city, model_name, model_dct, data, partition):
        logger.info("Start evaluationg on {} set".format(partition))
        result = []
        n = 1
        total = len(model_dct)
        for conf in model_dct.keys():
            logger.info("Evaluation progress: {}/{}".format(n, total))
            model = model_dct[conf]
            pred_res = model.predict(data)
            eval = ClassifierEvaluation(data.iloc[:,-1], pred_res)
            eval_res = eval.get_eval_result()
            result.append([
                conf,
                eval_res['accuracy'],
                eval_res['precision'],
                eval_res['recall'],
                eval_res['f1-score']
            ])
            n += 1
        result_df = pd.DataFrame(
            data=result,
            columns=['model_conf', 'accuracy', 'precision', 'recall', 'f1-score']
        )
        result_df.sort_values(
            by=['f1-score', 'recall', 'precision', 'accuracy'],
            ascending=[False, False, False, False],
            inplace=True
        )
        result_df.to_csv(
            self.eval_result_template.format(
                city,
                "{}_{}_{}".format(model_name, 'HyperparameterTuning', partition),
                index=False
            )
        )
        return result_df
    
    def save_trained_model(self, valid_result, model_dct):
        top_model = {}
        n = None
        if valid_result.shape[0] < self.top_model:
            n = valid_result.shape[0]
        else:
            n = self.top_model
        for i in range(n):
            row = valid_result.iloc[i,:]
            model = model_dct[row['model_conf']]
            model.save_model()
            top_model[row['model_conf']] = model
        return top_model

if __name__ == "__main__":
    exp = ClassifierHyperparameterExperiment()
    exp.run()