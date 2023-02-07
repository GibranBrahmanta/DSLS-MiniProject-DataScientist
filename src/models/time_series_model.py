import pandas as pd
import logging

logging.basicConfig(format='%(asctime)s [Time Series Model] %(message)s')
logger = logging.getLogger()
logger.setLevel(logging.INFO)

from pmdarima.arima import auto_arima
from statsmodels.tsa.arima.model import ARIMA

class Predictor:

    def __init__(self, city, model_name, attr) -> None:
        self.model_name = model_name
        self.attr = attr
        self.model = self.get_model(city, model_name, attr)
    
    def get_model(self, city, model_name, attr):
        model = None
        if model_name == 'ARIMA':
            model = ARIMAForecast(city, attr)
        elif model_name == 'SARIMA':
            model = SARIMAForecast(city, attr)
        return model
    
    def train(self, data):
        self.model.train(data)
    
    def predict(self, street, data, timestep):
        used_data = data.copy()
        used_data.index = pd.DatetimeIndex(used_data['time'], freq='H')
        used_data = used_data.drop(columns=['time'])
        pred_res = self.model.predict(street=street, data=used_data, timestep=timestep)
        return pred_res

class ARIMAForecast:

    param_path = "./model/time_series/ARIMA_{}_{}.parquet.gzip"

    def __init__(self, city, attr):
        try:
            self.city = city
            self.attr = attr
            self.param = pd.read_parquet(self.param_path.format(city, attr))
        except:
            logger.info("ARIMA model must be trained first")
    
    def train(self, data):
        lst_street = set(data['street'])

        result = []
        
        for street in lst_street:
            logger.info("Train on {}".format(street))
            used_data = data[data['street'] == street].loc[:,['time', self.attr]]
            used_data.index = pd.DatetimeIndex(used_data['time'], freq='H')
            used_data = used_data.drop(columns=['time'])
            arima_model = auto_arima(
                used_data, start_p=1, d=0, start_q=1, test='adf',
                max_p=6, max_d=5, max_q=6, m=1, seasonal=False,
                start_P=0, D=0, trace=False, error_action='ignore',
                suppress_warnings=False, stepwise=True
            )
            p, d, q = arima_model.get_params().get("order")
            result.append([
                street,
                p,
                d,
                q
            ])
        
        result_df = pd.DataFrame(data=result, columns=['street', 'p', 'd', 'q'])
        result_df.to_parquet(
            self.param_path.format(self.city, self.attr),
            index=False,
            compression="gzip"
        )

    def predict(self, street, data, timestep):
        p, d, q = self.get_param(street)
        model = ARIMA(data.values, order=(p,d,q))
        model_fitted = model.fit()
        res = model_fitted.forecast(steps=timestep)
        return res

    def get_param(self, street):
        data = self.param[self.param['street'] == street]
        return data['p'].values[0], data['d'].values[0], data['q'].values[0]


class SARIMAForecast:

    param_path = "./data/param/SARIMA_{}_{}.parquet.gzip"

    def __init__(self, city, attr):
        try:
            self.city = city
            self.attr = attr
            self.param = pd.read_parquet(self.param_path.format(city, attr))
        except:
            logger.info("SARIMA model must be trained first")
    
    def train(self, data):
        lst_street = set(data['street'])

        result = []
        
        for street in lst_street:
            logger.info("Train on {}".format(street))
            used_data = data[data['street'] == street].loc[:,['time', self.attr]]
            used_data.index = pd.DatetimeIndex(used_data['time'], freq='H')
            used_data = used_data.drop(columns=['time'])
            arima_model = auto_arima(
                used_data, start_p=1, d=0, start_q=1, test='adf',
                max_p=6, max_d=5, max_q=6, m=6, seasonal=True,
                start_P=1, D=0, start_Q=1, max_P=6, max_D=5, max_Q=6,
                trace=False, error_action='ignore',
                suppress_warnings=False, stepwise=True
            )
            p, d, q = arima_model.get_params().get("order")
            s_p, s_d, s_q, s_m = arima_model.get_params().get("seasonal_order")
            result.append([
                street,
                p,
                d,
                q,
                s_p,
                s_d,
                s_q,
                s_m
            ])
        
        result_df = pd.DataFrame(data=result, columns=['street', 'p', 'd', 'q', 's_p', 's_d', 's_q', 's_m'])
        result_df.to_parquet(
            self.param_path.format(self.city, self.attr),
            index=False,
            compression="gzip"
        )



