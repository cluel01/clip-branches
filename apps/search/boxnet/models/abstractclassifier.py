from abc import ABC, abstractmethod
 
class AbstractClassifier(ABC):
    @abstractmethod
    def fit(self, X, y):
        pass
    
    @abstractmethod
    def predict(self, X, y):
        pass
    
    @abstractmethod
    def set_hyperparameters(self, hyperparameters:dict):
        pass
    
    @abstractmethod
    def get_num_boxes(self):
        pass