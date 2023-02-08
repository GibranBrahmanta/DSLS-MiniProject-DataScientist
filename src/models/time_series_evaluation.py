from sklearn.metrics import mean_absolute_error, mean_squared_error

class Evaluation:
    
    def __init__(self, actual, pred):
        self.actual = actual
        self.pred = pred
    
    def get_rmse(self):
        return mean_squared_error(self.actual, self.pred, squared=False)
    
    def get_mae(self):
        return mean_absolute_error(self.actual, self.pred)
        
    def get_eval_result(self):
        result = {
            'rmse': self.get_rmse(),
            'mae': self.get_mae()
        }
        return result