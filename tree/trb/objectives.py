import abc
import imp
from sklearn.metrics import roc_auc_score, r2_score
import numpy as np


class Objective(abc.ABC):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def loss(self, labels, logits):
        pass

    @abc.abstractmethod
    def gradients(self, labels, logits):
        pass

    @abc.abstractmethod
    def hessians(self, labels, predictions):
        pass

    


class BinaryCrossentropy(Objective):

    def loss(self, labels, probs):
        loss = -(labels * np.log(probs) + (1.0 - labels) * np.log(1.0 - probs))
        return np.mean(loss)

    def gradients(self, labels, probs):
        # Derivative of -(y * log(1/(1+exp(-w))) + (1.0 - y) * log(1.0 - 1/(1+exp(-w)))) wrt w
        # probs = 1/(1+exp(-w))
        return probs - labels  # * 1/len(labels)

    def hessians(self, labels, probs):
        return probs * (1 - probs)  # * 1/len(labels)
        # return np.full(len(labels), 0.25)  # * 1/len(labels)

    def transformation(self, x):
        return 1.0 / (1.0 + np.exp(-x))
    
    def inverse_trans(sefl, y):
        return np.log(y/(1-y))
    
    def metric(self, labels, probs):
        return roc_auc_score(labels, probs)
        

    
class SquaredError(Objective):
    
    def loss(self, labels, predictions):
        # In order to compare the loss of different datasets the mean is computed
        return np.mean(np.square(labels - predictions))
        # return np.mean(np.square(labels - predictions))

    def gradients(self, labels, predictions):
        # XGBoost doesn't multiply with 1/len(labels)
        return -2*(labels - predictions)  # * 1/len(labels)

    def hessians(self, labels, predictions):
        # XGBoost doesn't multiply with 1/len(labels)
        return np.full(len(labels), 2)  # * 1/len(labels)

    def transformation(self, x):
        return x
    
    def inverse_trans(sefl, y):
        return y

    def metric(self, labels, predictions):
        return r2_score(labels, predictions)
    
        
class MAE(Objective):

    def loss(self, labels, predictions):
        # In order to compare the loss of different datasets the mean is computed
        return np.mean(np.abs(labels - predictions))
        # return np.mean(np.square(labels - predictions))

    def gradients(self, labels, predictions):
        return np.sign(predictions-labels)

    def hessians(self, labels, predictions):
        return np.full(len(labels), 0.0)  # * 1/len(labels)
        # return np.full(len(labels), 1)  # * 1/len(labels)

    def transformation(self, x):
        return x
    
    def inverse_trans(sefl, y):
        return y
    
    def metric(self, labels, predictions):
        return r2_score(labels, predictions)

class SmoothL1(Objective):

    def loss(self, labels, predictions, beta=0.5):
        # In order to compare the loss of different datasets the mean is computed
        self.beta = beta
        residual = np.abs(predictions - labels)
        maskin = np.where(residual < self.beta)[0]
        maskout = np.where(residual >= self.beta)[0]
        loss = np.zeros(labels.shape[0])
        loss[maskin] = 0.5 * np.square(residual[maskin]) / self.beta
        loss[maskout] = residual[maskout] - 0.5 * self.beta
        return np.mean(loss)
        
    def gradients(self, labels, predictions):
        # XGBoost doesn't multiply with 1/len(labels)
        residual = predictions - labels
        maskin = np.where(np.abs(residual) < self.beta)[0]
        maskout = np.where(np.abs(residual) >= self.beta)[0]
        g = np.zeros(labels.shape[0])
        g[maskin] = residual[maskin] / self.beta
        g[maskout] = np.where(residual[maskout]>0,1,-1)
        return g  # * 1/len(labels)

    def hessians(self, labels, predictions):
        # XGBoost doesn't multiply with 1/len(labels)
        residual = predictions - labels
        maskin = np.where(np.abs(residual) < self.beta)[0]
        maskout = np.where(np.abs(residual) >= self.beta)[0]
        h = np.zeros(labels.shape[0])
        h[maskin] = 1 / self.beta
        h[maskout] = 0
        return h  # * 1/len(labels)

    def transformation(self, x):
        return x
    
    def inverse_trans(sefl, y):
        return y
    
    def metric(self, labels, predictions):
        return r2_score(labels, predictions)
    

class Huber(Objective):
    
    def loss(self, labels, predictions, beta=1):
        # In order to compare the loss of different datasets the mean is computed
        self.beta = beta
        residual = np.abs(predictions - labels)
        maskin = np.where(residual < self.beta)[0]
        maskout = np.where(residual >= self.beta)[0]
        loss = np.zeros(labels.shape[0])
        loss[maskin] = 0.5 * np.square(residual[maskin]) 
        loss[maskout] = beta*(residual[maskout] - 0.5 * self.beta)
        return np.mean(loss)
        
    def gradients(self, labels, predictions):
        # XGBoost doesn't multiply with 1/len(labels)
        residual = predictions - labels
        maskin = np.where(np.abs(residual) < self.beta)[0]
        maskout = np.where(np.abs(residual) >= self.beta)[0]
        g = np.zeros(labels.shape[0])
        g[maskin] = residual[maskin] 
        g[maskout] = self.beta*np.where(residual[maskout]>0,1,-1)
        return g  # * 1/len(labels)

    def hessians(self, labels, predictions):
        # XGBoost doesn't multiply with 1/len(labels)
        residual = predictions - labels
        maskin = np.where(np.abs(residual) < self.beta)[0]
        maskout = np.where(np.abs(residual) >= self.beta)[0]
        h = np.zeros(labels.shape[0])
        h[maskin] = 1 
        h[maskout] = 0
        return h  # * 1/len(labels)

    def transformation(self, x):
        return x
    
    def inverse_trans(sefl, y):
        return y
    
    def metric(self, labels, predictions):
        return r2_score(labels, predictions)
    

class L12(Objective):
    def __init__(self, beta=1):
        self.beta = beta
        
    def loss(self, labels, predictions):
        residual = predictions - labels
        loss = (np.abs(residual)<self.beta)*0.5*residual**2+(np.abs(residual)>=1)*(2*np.abs(residual)**0.5*self.beta**(1.5)-1.5*self.beta**2)
        return np.mean(loss)
        
    def gradients(self, labels, predictions):
        residual = predictions - labels
        return residual*(np.abs(residual)<self.beta)+\
            (np.abs(residual)>=self.beta)*(np.divide(np.ones(residual.shape)*self.beta**(1.5),
                                                     np.sign(residual)*np.abs(residual)**(0.5),
                                                     out=np.zeros_like(residual), where=(residual!=0)))

    def hessians(self, labels, predictions):
        residual = predictions - labels
        return 1*(np.abs(residual)<self.beta)+\
            (np.abs(residual)>=self.beta)*(np.divide(-np.ones(residual.shape)*self.beta**(1.5),
                                                     2*np.abs(residual)**(1.5),
                                                     out=np.zeros_like(residual), where=(residual!=0))) 

    def transformation(self, x):
        return x
    
    def inverse_trans(sefl, y):
        return y
    
    def metric(self, labels, predictions):
        return r2_score(labels, predictions)


    
    