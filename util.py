import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score, matthews_corrcoef, confusion_matrix, roc_curve

# Preprocessing
class MaxImputer(TransformerMixin):
    def fit(self, X, y=None):
        self.max_values = np.nanmax(X, axis=0, keepdims=True)
        return self

    def transform(self, X):
        X = np.where(np.isnan(X), self.max_values, X)
        return X

class OutlierHandler(BaseEstimator, TransformerMixin):
    def __init__(self, percentile=0.25):
        self.percentile = percentile
        
    def fit(self, X, y=None):
        X = pd.Series(X.flatten())
        Q1 = X.quantile(self.percentile)
        Q3 = X.quantile(1-self.percentile)
        IQR = Q3 - Q1
        self.lb = Q1 - 1.5 * IQR
        self.ub = Q3 + 1.5 * IQR
        return self
    
    def transform(self, X):
        X = pd.Series(X.flatten())
        X = np.where(X < self.lb, self.lb, X)
        X = np.where(X > self.ub, self.ub, X)
        return X.reshape(-1,1)

# Evaluation
def predict(model, X_test):
    y_score = model.predict_proba(X_test)[:,1]    
    return y_score

def get_ks_cutoff(y_test, y_score):
    fpr, tpr, thresholds = roc_curve(y_test, y_score)
    ks_values = tpr - fpr  # KS = max(TPR - FPR)
    optimal_idx = np.argmax(ks_values)
    cutoff = thresholds[optimal_idx]
    return(cutoff)

def evaluate(model, y_test, X_test=None, y_score=None, cutoff=None, sample_weight=None):
    assert (y_score is not None) or (X_test is not None), 'Please input y_score or X_test'
    if X_test is not None:
        y_score = predict(model, X_test)

    if cutoff is None:
        cutoff = get_ks_cutoff(y_test, y_score)

    y_pred = (y_score >= cutoff).astype(int)
    res = {'AUC' : round(roc_auc_score(y_test, y_score, sample_weight=sample_weight), 5),
         'f1_score' : round(f1_score(y_test, y_pred, sample_weight=sample_weight), 5),
         'precision' : round(precision_score(y_test, y_pred, sample_weight=sample_weight), 5),
         'recall' : round(recall_score(y_test, y_pred, sample_weight=sample_weight), 5),
         'accuracy' : round(accuracy_score(y_test, y_pred, sample_weight=sample_weight), 5),
         'mcc' : round(matthews_corrcoef(y_test, y_pred, sample_weight=sample_weight), 5),
         'confusion_matrix' : confusion_matrix(y_test, y_pred, sample_weight=sample_weight),
                } 
    return res, y_score, cutoff