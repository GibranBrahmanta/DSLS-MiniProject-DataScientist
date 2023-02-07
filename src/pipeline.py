import logging
import sys
import os

sys.path.append(os.getcwd())

logging.basicConfig(format='%(asctime)s [Pipeline] %(message)s')
logger = logging.getLogger()
logger.setLevel(logging.INFO)

from data.make_dataset import DatasetGenerator
from features.build_features import FeatureEngineer

class Pipeline:

    def __init__(self) -> None:
        self.dataset_generator = DatasetGenerator()
        self.feature_engineer = FeatureEngineer()

    def run(self):
        self.dataset_generator.create_dataset()
        self.feature_engineer.build_feature()
        




