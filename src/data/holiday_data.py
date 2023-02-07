import logging
import pandas as pd
import requests
import sys
import os

sys.path.append(os.getcwd())

logging.basicConfig(format='%(asctime)s [Get Holiday Data] %(message)s')
logger = logging.getLogger()
logger.setLevel(logging.INFO)

from bs4 import BeautifulSoup
from datetime import datetime
from src.utils.config import config

dataset = config['dataset']

class HolidayScrapper:

    url_template = "https://excelnotes.com/holidays-indonesia-{}/"

    def __init__(self, year, path) -> None:
        self.year = year
        self.url = self.url_template.format(self.year)
        self.path = path.format(year)

    def create_dataset(self) -> pd.DataFrame:
        logger.info("Start process")
        try:
            content = self.get_web_content()
            data = self.get_holiday_data(content)
            formatted_data = self.reformat_data(data)
            dataset = self.save_dataset(formatted_data)
            logger.info("Finish process")
            return dataset
        except Exception as e:
            logger.info("ERROR: {} - {}".format(e.__class__.__name__, str(e)))
            return None

    def get_web_content(self) -> str:
        page = requests.get(self.url)
        logger.info("Finish get web content")
        return page.content
    
    def get_holiday_data(self, content) -> list:
        parsed_content = BeautifulSoup(content, "html.parser")
        holiday_data = parsed_content.find("table")
        result = [
            self.extract_row(row) for row in holiday_data.find_all("tr") if "Note" not in row.text
        ]
        logger.info("Finish get holiday data")
        return result
    
    def extract_row(self, data) -> list:
        all_columns = data.find_all("td")
        result = [all_columns[i].text for i in range(2)]
        return result
    
    def reformat_data(self, data) -> list:
        result = [
            [row[0], self.change_date_format(row[1])] for row in data
        ]
        logger.info("Finish reformat data")
        return result
    
    def change_date_format(self, data) -> datetime:
        old_format = '%b %d, %Y'
        new_format = '%Y-%m-%d %H:%M:%S.%f'
        converted_data = datetime.strptime(data, old_format)
        return datetime.strptime(
            datetime.strftime(converted_data, new_format), 
            new_format
        )
    
    def save_dataset(self, data) -> pd.DataFrame:
        columns = [
            "holiday",
            "date"
        ]
        df = pd.DataFrame(data=data, columns=columns)
        df.to_parquet(self.path, 
                    index=False, compression="gzip")
        logger.info("Finish save data")
        return df
