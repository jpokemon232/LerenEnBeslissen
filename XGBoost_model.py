import xgboost as xgb
from util import DataHandler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
L_data = pd.read_csv('H:\\project\\ND-LerenEnBeslissen\\stedin-dataset\\Middelharnis.csv')
print(np.unique(L_data['locatie']))
print(np.unique(L_data['spanning(kV)']))
print(np.unique(L_data['transformator']))
params = {'n_estimators': 318, 'grow_policy': 'lossguide', 'tree_method': 'hist', 'max_depth': 3, 'min_child_weight': 0.3867219450425303, 'subsample': 0.9446541028519908, 'colsample_bynode': 0.34797604051723946, 'reg_lambda': 3.177784399546179e-05, 'learning_rate': 0.016035447591742306, 'gamma': 7.059667726759788e-05, 'reg_alpha': 0.011148300861008344}

train_percentage = 0.7
for loc in np.unique(L_data['locatie']):
    for span in np.unique(L_data['spanning(kV)']):
        for trans in np.unique(L_data['transformator']):
            try:
                length = 36
                num = 0
                new_data = []
                DH = DataHandler(loc + '.csv',trans,span)
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

                model = xgb.XGBRegressor(**params,early_stopping_rounds=50)
                model.fit(X_train,Y_train,eval_set=[(X_test,Y_test)],verbose=0)

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
                print(loc,trans,span)
                print(score)
            except:
                pass
# zit nog een fout in, de True en Predicted plots hebben een lagg van 32, dus de predicted lijn moet 32 naar links worden geschoven
plt.plot(data.index,data['number_tap_changes'],label='True')
plt.plot(data.index,output_pred,label='predict')
plt.legend()
plt.show()