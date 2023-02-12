from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score 

class ClassifierEvaluation:

    def __init__(self, actual, pred):
        self.actual = actual
        self.pred = pred

    def get_accuracy(self):
        return accuracy_score(self.actual, self.pred)
    
    def get_precision(self):
        return precision_score(self.actual, self.pred, average='macro')

    def get_recall(self):
        return recall_score(self.actual, self.pred, average='macro')
    
    def get_f1_score(self):
        return f1_score(self.actual, self.pred, average='macro')
    
    def get_eval_result(self):
        result = {
            'accuracy': self.get_accuracy(),
            'precision': self.get_precision(),
            'recall': self.get_recall(),
            'f1-score': self.get_f1_score()
        }
        return result