import pickle

from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier

class Classifier:

    model_path = "./model/classifier/{}.pkl"

    def __init__(self, city, model_name, param=None) -> None:
        self.city = city
        self.model_conf = self.get_model_conf(city, model_name, param)
        self.model = self.create_model(model_name, param)
    
    def get_model_conf(self, city, model_name, param) -> str:
        res = "{}_{}_{}"
        if param:
            conf = ""
            for key in param.keys():
                conf += "({}_{})".format(key, str(param[key]))
        else:
            conf = "baseline"
        return res.format(city, model_name, conf)

    def create_model(self, model_name, param) -> object:
        if param:
            return self.create_model_with_param(model_name, param)
        else:
            return self.create_baseline_model(model_name)
    
    def create_baseline_model(self, model_name) -> object:
        model = None
        if model_name == 'LogisticRegression':
            model = LogisticRegression()
        elif model_name == 'SVM':
            model = LinearSVC()
        elif model_name == 'NaiveBayes':
            model = ComplementNB()
        elif model_name == 'DecisionTree':
            model = DecisionTreeClassifier()
        elif model_name == 'RandomForest':
            model = RandomForestClassifier()
        elif model_name == 'LightGBM':
            model = LGBMClassifier()
        elif model_name == 'XGBoost':
            model = XGBClassifier()
        return model
    
    def create_model_with_param(self, model_name, param) -> object:
        model = None
        if model_name == 'LogisticRegression':
            model = LogisticRegression(**param)
        elif model_name == 'SVM':
            model = LinearSVC(**param)
        elif model_name == 'NaiveBayes':
            model = ComplementNB(**param)
        elif model_name == 'DecisionTree':
            model = DecisionTreeClassifier(**param)
        elif model_name == 'RandomForest':
            model = RandomForestClassifier(**param)
        elif model_name == 'LightGBM':
            model = LGBMClassifier(**param)
        elif model_name == 'XGBoost':
            model = XGBClassifier(**param)
        return model
    
    def train(self, data) -> None:
        X = data.iloc[:,0:-1]
        y  = data.iloc[:,-1]
        if "NaiveBayes" in self.model_conf:
            X.replace(-1, 8, inplace=True)
        self.model.fit(X, y)
    
    def predict(self, data) -> list:
        return self.model.predict(data.iloc[:,0:-1])
    
    def save_model(self) -> None:
        path = self.model_path.format(self.model_conf)
        with open(path, 'wb') as file:
            pickle.dump(self.model, file)