# Need to adapt this based on the model above
from abc import ABCMeta, abstractmethod

# Create abstract framework for building models
class AbstractModelWrapper(metaclass=ABCMeta):
    """ Abstract class for deep learning models. """
    
    @abstractmethod
    def train(self):
        """ Train model. """
        pass

    @abstractmethod
    def predict(self):
        """ Predict from input data."""
        pass
        
    @abstractmethod
    def print(self):
        """ Print a representation of the model. """
        pass
    
    @abstractmethod
    def _load_weights(self, filename):
        """ Load weights from file. """
        pass
        
    @abstractmethod
    def _save_weights(self, filename):
        """ Load weights from file. """
        pass
        
    @abstractmethod
    def _build_model(self):
        """ Build and compile a model. """
        pass
    
        
    


