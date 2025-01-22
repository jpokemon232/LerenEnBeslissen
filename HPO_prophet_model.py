import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from util import DataHandler
from prophet import Prophet
import optuna
import logging
import sys


def objective(trial):
    param = {
        "growth": trial.suggest_categorical('growth',['linear','logistic', 'flat']),
        "n_changepoints": trial.suggest_int('n_changepoints',0,100),
        "yearly_seasonality": trial.suggest_categorical('yearly_seasonality',[True,False]),
        "weekly_seasonality": trial.suggest_categorical('weekly_seasonality',[True,False]),
        "daily_seasonality": trial.suggest_categorical('daily_seasonality',[True,False]),
        "seasonality_mode": trial.suggest_categorical('seasonality_mode',['additive','multiplicative']),
        "seasonality_prior_scale": trial.suggest_float('seasonality_prior_scale',0,100),
        "holidays_prior_scale": trial.suggest_float('holidays_prior_scale',0,100),
        "changepoint_prior_scale": trial.suggest_float('changepoint_prior_scale',0,1),
        "mcmc_samples": trial.suggest_int('mcmc_samples',0,100),
        "interval_width": 0.8,
        "uncertainty_samples": 1000
    }
    try:
        scores = []
        for _ in range(5):
            m = Prophet(**param)
            m.fit(train_data)
            #calculating the test losses
            prediction = m.predict(test_data)

            #through an relu
            prediction['yhat'] = prediction['yhat'] * (prediction['yhat'] > 0)
            prediction['yhat_lower'] = prediction['yhat_lower'] * (prediction['yhat_lower'] > 0)

            score = np.mean(np.abs(np.array(test_data['y']) - np.array(prediction['yhat'])))
            scores.append(score)
        return np.mean(scores)
    except:
        return np.inf


if __name__ == '__main__':
    train_percentage = 0.7

    DH = DataHandler('Dordrecht.csv','TR2','50-13')
    data = DH.return_data()

    m = Prophet(
        weekly_seasonality=True,
        daily_seasonality=True
    )
    data['hist_timestamp'] = data.index
    data = data.copy()
    data.rename(columns={'number_tap_changes':'y','hist_timestamp':'ds'},inplace=True)

    train_data, test_data = data.iloc[:int(len(data)*train_percentage)],data.iloc[int(len(data)*train_percentage):]
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    study_name = "prophet_study"  # Unique identifier of the study.
    storage_name = "sqlite:///HPO//{}.db".format(study_name)
    study = optuna.create_study(study_name=study_name, storage=storage_name)
    study.optimize(objective, n_trials=250)