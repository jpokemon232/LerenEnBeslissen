import xgboost as xgb
from util import DataHandler
import numpy as np
import optuna
import logging
import sys
import os

def objective(trial):

    params = {
        'n_estimators': trial.suggest_int('n_estimators',1,4000),
        'grow_policy': trial.suggest_categorical('grow_policy',['depthwise','lossguide']),
        'tree_method': trial.suggest_categorical('tree_method', ['approx', 'hist']),
        'max_depth': trial.suggest_int('max_depth', 1, 5),
        'min_child_weight': trial.suggest_float('min_child_weight', 0, 2),
        'subsample': trial.suggest_float('subsample', 0.1, 0.7),
        'colsample_bynode': trial.suggest_float('colsample_bynode', 0.1, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.001, 25, log=True),
        'learning_rate': trial.suggest_float('learning_rate', 0.08, 1, log=True),
        'gamma':trial.suggest_float('gamma', 0.001, 25, log=True),
        'reg_alpha':trial.suggest_float('reg_alpha', 0.001, 25, log=True)
    }

    model = xgb.XGBRegressor(**params,early_stopping_rounds=50)
    model.fit(X_train,Y_train,eval_set=[(X_test,Y_test)],verbose=0)
    score = model.score(X_test,Y_test)
    if len(study.trials) != 1:
        if score > study.best_value:
            model.save_model(os.path.dirname(os.path.abspath(__file__)) + "\\xgboost_models\\xgboost_model_" + str(score) + ".json")
    return score
    

if __name__ == '__main__':
    train_percentage = 0.7

    DH = DataHandler('Dordrecht.csv','TR2','50-13')
    data = DH.return_data()

    num = 0
    new_data = []
    for index in range(len(data)):
        num += 1
        if data['jumps'].iloc[index] == 1:
            new_data.append(num)
            num = 0

    X = []
    Y = []
    for index in range(len(new_data)-32):
        X.append(new_data[index:index+32])
        Y.append(new_data[index+32])

    X = np.array(X)
    Y = np.array(Y)
    X_train, Y_train,X_test,Y_test = X[:int(len(X)*train_percentage)],Y[:int(len(Y)*train_percentage)],X[int(len(X)*train_percentage):],Y[int(len(Y)*train_percentage):]
    
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    study_name = "xgboost_study_v14"  # Unique identifier of the study.
    storage_name = "sqlite:///HPO//{}.db".format(study_name)
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        sampler=optuna.samplers.TPESampler(n_startup_trials=1000),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10,n_warmup_steps=1000,interval_steps=2,n_min_trials=4),
        direction='maximize'
    )
    study.optimize(objective, n_trials=5000,show_progress_bar=True)
    #v1: n_startup_trials=100, n_startup_trials=100
    #v2: n_startup_trials=10, n_startup_trials=10,n_warmup_steps=100 (really good)
    #v4: n_startup_trials=10, n_startup_trials=100,n_warmup_steps=10
    #v5: n_startup_trials=100, n_startup_trials=100,n_warmup_steps=10
    #v6: n_startup_trials=100, n_startup_trials=10,n_warmup_steps=100
    #v7: n_startup_trials=10, n_startup_trials=10,n_warmup_steps=100 