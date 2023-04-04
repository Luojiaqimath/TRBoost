import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import data_reg
from trb import objectives
from trb.learner import TRBLearner
import xgboost as xgb
from sklearn.ensemble import GradientBoostingRegressor
import warnings
warnings.filterwarnings('ignore')


folder_path = "./reg_gradient_result"
if not os.path.exists(folder_path):
    os.makedirs(folder_path)


objective = objectives.SquaredError()
n_estimators = 100

for dataset in ['concrete',  'energy', 'power', 'kin8nm', 'wine_quality', 'california']:
    print('data: ' + dataset)
    X, y = data_reg.loader(dataset)

    trb_best_loss_list = []
    trb_best_r2_list = []
    trb_best_nestimator_list = []

    for i in range(5):
        np.random.seed(i)
        seed = np.random.randint(10000, size=1).item()
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=seed)
        train_x, eval_x, train_y, eval_y = train_test_split(
            X_train, y_train, test_size=0.2, random_state=seed)

        # Trb grid search
        trb_best_nestimator = n_estimators
        trb_best_eval_loss = 100
        trb_best_alpha = 0.1
        trb_best_eta = 0
        alpha_list = [0.1, 0.5, 1, 5]
        eta_list = [0, 0.01, 0.1]
        for _alpha in alpha_list:
            for _eta in eta_list:
                trb_reg = TRBLearner(objective,
                                    base_score=np.mean(train_y), 
                                        n_estimators=n_estimators,
                                        alpha=_alpha,
                                        eta=_eta,
                                        update_strategy='gradient')
                trb_reg.fit(train_set=(train_x, train_y), eval_set=(eval_x, eval_y))
                trb_eval_loss = trb_reg.eval_loss_list
                if np.nanmin(trb_eval_loss) < trb_best_eval_loss:
                    trb_best_nestimator = np.nanargmin(trb_eval_loss)+1
                    trb_best_alpha = _alpha
                    trb_best_eta = _eta
                    trb_best_eval_loss = np.nanmin(trb_eval_loss)

        trb_reg = TRBLearner(objective,
                     base_score=np.mean(y_train),
                        n_estimators=trb_best_nestimator,
                        alpha=trb_best_alpha,
                        eta=trb_best_eta,
                        update_strategy='gradient')
        trb_reg.fit(train_set=(X_train, y_train))
        y_trb_pred = trb_reg.predict(X_test)
        trb_r2 = r2_score(y_test, y_trb_pred) 
        trb_loss = mean_squared_error(y_test, y_trb_pred)
        trb_best_r2_list.append(trb_r2)   
        trb_best_nestimator_list.append(trb_best_nestimator)
        trb_best_loss_list.append(trb_loss)
        
        # Saving...
        current_path = os.path.join(folder_path, dataset, f'shuffle_{seed}', 'trb', 'evaluate')

        if not os.path.exists(current_path):
            os.makedirs(current_path)

        np.savez(os.path.join(current_path, "data.npz"), 
            loss = trb_loss,
            metric = trb_r2,
            alpha = trb_best_alpha,
            eta = trb_best_eta,
            n_estiamtors = trb_best_nestimator,
        )

    np.savez(os.path.join(folder_path, dataset, 'metric_lists.npz'),
        trb_r2 = trb_best_r2_list,
        trb_nestimator = trb_best_nestimator_list,
        trb_loss = trb_best_loss_list,
    )
    print('-'*10)
