from .. import io
import os
from yellowcab.cabana import *
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.linear_model import SGDRegressor, SGDClassifier
import sklearn.metrics as sm
from math import radians
from .transform_nyc import *

class model:
    def __init__(self,X=None):
        self.X = transform_nyc(X)

    def transform(self,):
        return self.X

    def predict_trip_distance(self):
        X = self.X
        ya = X['trip_distance']
        Xa = X.drop(
            ['trip_distance', 'pd', 'duration', 'passenger_count', 'payment_type', 'pay2', 'pay3', 'pay4', 'r2',
             'r3', 'r4', 'r5', 'r6',
             'fare_amount', 'extra', 'mta_tax', 'tip_amount', 'tolls_amount',
             'improvement_surcharge', 'total_amount', 'congestion_surcharge',
             'DOweekend', 'DOhoursin', 'DOhourcos', 'DOdaysin', 'DOdaycos', 'DOmonthsin', 'DOmonthcos'], axis=1)
        Xtrain, Xtest, ytrain, ytest = train_test_split(Xa, ya, test_size=0.2, random_state=0)
        modelhya = SGDRegressor(random_state=0)
        hyperparameters = {'loss': ['squared_loss', 'huber', 'epsilon_insensitive'], 'alpha': np.linspace(0, 0.001, 3)}
        l = sorted(sm.SCORERS.keys())
        grid= GridSearchCV(modelhya, hyperparameters, cv=5, scoring='neg_mean_absolute_error',n_jobs=-1 )
        grid.fit(Xtrain, ytrain)
        comparison = pd.DataFrame(np.abs(grid.cv_results_['mean_test_score']), columns=['Loss'],
                                  index=['squared0', 'huber0', 'ep0', 'squared0.0005', 'huber0.0005', 'ep.0005',
                                         'squared0.001', 'huber0.001', 'ep0.001'])
        modela = SGDRegressor(alpha=0, random_state=0)
        modela.fit(Xtrain, ytrain)
        predicted_distance = modela.predict(X)
        # io.save_model(modela)

    def predict_fare_amount(self):
        pass

    def predict_payment_type(self):
        pass

    def predict(self):
        pass




        # def train():
    #     lin = LinearRegression()
    #     print("Linear model created")
    #     print("Training...")

