import logging
import sys
import os

sys.path.append(os.getcwd())

logging.basicConfig(format='%(asctime)s [Pipeline] %(message)s')
logger = logging.getLogger()
logger.setLevel(logging.INFO)

from data.make_dataset import DatasetGenerator
from features.build_features import FeatureEngineer
from models.classifier_baseline import ClassifierBaselineExperiment
from models.classifier_hyperparameter_tuning import ClassifierHyperparameterExperiment
from models.train_evaluate_time_series import TimeSeriesExperiment

class Pipeline:

    def __init__(self) -> None:
        self.dataset_generator = DatasetGenerator()
        self.feature_engineer = FeatureEngineer()
        self.time_series_experiment = TimeSeriesExperiment()
        self.classifier_baseline_experiment = ClassifierBaselineExperiment()
        self.classifier_hyperparameter_experiment = ClassifierHyperparameterExperiment()

    def run(self):
        self.dataset_generator.create_dataset()
        self.feature_engineer.build_feature()
        self.time_series_experiment.run()
        self.classifier_baseline_experiment.run()
        self.classifier_hyperparameter_experiment.run()

if __name__ == '__main__':
    pipeline = Pipeline()
    pipeline.run()
        




