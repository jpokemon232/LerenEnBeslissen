import xgboost as xgb
from util import DataHandler
import numpy as np
import optuna
import logging
import sys
import os

def objective(trial):
    length = trial.suggest_int('length',8,128)

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
    for index in range(len(new_data)-length):
        X.append(new_data[index:index+length])
        Y.append(new_data[index+length])

    X = np.array(X)
    Y = np.array(Y)
    X_train, Y_train,X_test,Y_test = X[:int(len(X)*train_percentage)],Y[:int(len(Y)*train_percentage)],X[int(len(X)*train_percentage):],Y[int(len(Y)*train_percentage):]

    params = {
        'n_estimators': trial.suggest_int('n_estimators',1,6000),
        'grow_policy': trial.suggest_categorical('grow_policy',['depthwise','lossguide']),
        'tree_method': trial.suggest_categorical('tree_method', ['approx', 'hist']),
        'max_depth': trial.suggest_int('max_depth', 1, 5),
        'min_child_weight': trial.suggest_float('min_child_weight', 0.01, 2, log=True),
        'subsample': trial.suggest_float('subsample', 0.01, 1),
        'colsample_bynode': trial.suggest_float('colsample_bynode', 0.1, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 10**-5, 1, log=True),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 2, log=True),
        'gamma':trial.suggest_float('gamma', 10**-5, 1, log=True),
        'reg_alpha':trial.suggest_float('reg_alpha', 10**-5, 1, log=True)
    }
    model = xgb.XGBRegressor(**params,early_stopping_rounds=50)
    model.fit(X_train,Y_train,eval_set=[(X_test,Y_test)],verbose=0)
    #score = model.score(X_test,Y_test)
    #score = np.mean(np.abs(model.predict(X_test) - Y_test))
    running_X = X_test[-1]
    Y_pred = []
    for _ in range(len(X_test)):
        pred_1 = int(list(model.predict(np.array([running_X])))[0])
        Y_pred.append(pred_1)
        running_X = np.array(list(running_X[1:]) + [pred_1])
    
    index = 0
    output_pred = np.array([data['number_tap_changes'].iloc[-sum(Y_test)]]*sum(Y_test))
    for p in range(len(Y_pred)):
        output_pred[index:] += 1
        index += int(Y_pred[p])
    output_pred = output_pred[:index]
    score = np.mean(np.abs(output_pred - data['number_tap_changes'].iloc[-sum(Y_test):].iloc[:index]))

    if len(study.trials) != 1:
        if np.log(score) < study.best_value:
            model.save_model(os.path.dirname(os.path.abspath(__file__)) + "\\xgboost_models\\xgboost_model_" + str(score) + ".json")
    return np.log(score) #because i want to see more clearly
    

if __name__ == '__main__':
    train_percentage = 0.7
    
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    study_name = "xgboost_study_v34"  # Unique identifier of the study.
    storage_name = "sqlite:///HPO//{}.db".format(study_name)
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        sampler=optuna.samplers.TPESampler(n_startup_trials=300),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10,n_warmup_steps=300)
    )
    study.optimize(objective, n_trials=1500,show_progress_bar=True)
    # length = 32 for v1 - v15, length = 4 is better
    #v1: n_startup_trials=100, n_startup_trials=100
    #v2: n_startup_trials=10, n_startup_trials=10,n_warmup_steps=100 (really good)
    #v4: n_startup_trials=10, n_startup_trials=100,n_warmup_steps=10
    #v5: n_startup_trials=100, n_startup_trials=100,n_warmup_steps=10
    #v6: n_startup_trials=100, n_startup_trials=10,n_warmup_steps=100
    #v7: n_startup_trials=10, n_startup_trials=10,n_warmup_steps=100 
    # v3 is made with RMSE loss of test data, using model.best_value directly
    # v13 n_startup_trials=1000, n_startup_trials=10,n_warmup_steps=1000