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



def gbdt_reg_score(reg, X_test, y_test, n_estimators):
    loss = np.zeros((n_estimators,), dtype=np.float64)
    score = np.zeros((n_estimators,), dtype=np.float64)
    for i, y_pred in enumerate(reg.staged_predict(X_test)):
        loss[i] = reg.loss_(y_test, y_pred)
        score[i] = r2_score(y_test, y_pred)
    return loss, score


folder_path = "./reg_standard_result"
if not os.path.exists(folder_path):
    os.makedirs(folder_path)


objective = objectives.SquaredError()
n_estimators = 100


for dataset in ['concrete',  'energy', 'power', 'kin8nm', 'wine_quality', 'california']:
    print('data: ' + dataset)
    X, y = data_reg.loader(dataset)

    trb_best_loss_list = []
    xgb_best_loss_list = []
    gbdt_best_loss_list = []

    trb_best_r2_list = []
    xgb_best_r2_list = []
    gbdt_best_r2_list = []

    trb_best_nestimator_list = []
    xgb_best_nestimator_list = []
    gbdt_best_nestimator_list = []

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
                                        eta=_eta)
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
                        eta=trb_best_eta)
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
            r2 = trb_r2,
            alpha = trb_best_alpha,
            eta = trb_best_eta,
            n_estiamtors = trb_best_nestimator,
        )

        # xgb
        lr_list = [0.1, 0.5, 1]
        lambd_list = [0, 0.5, 1, 5]
        xgb_best_nestimator = n_estimators
        xgb_best_eval_loss = 100
        xgb_best_lr = 0.1
        xgb_best_lambd = 0
        for lr in lr_list:
            for lambd in lambd_list:
                xgb_reg = xgb.XGBRegressor(n_estimators=n_estimators, 
                                                tree_method='exact',
                                                learning_rate=lr,
                                                reg_lambda=lambd)
                xgb_reg.fit(train_x, train_y)
                xgb_eval_loss = []
                for i in range(n_estimators):
                    y_eval_pred = xgb_reg.predict(eval_x, iteration_range=(0, i+1))
                    xgb_eval_loss.append(objective.loss(eval_y, y_eval_pred))

                if np.nanmin(xgb_eval_loss) < xgb_best_eval_loss:
                    xgb_best_nestimator = np.nanargmin(xgb_eval_loss)+1
                    xgb_best_lr = lr
                    xgb_best_lambd = lambd
                    xgb_best_eval_loss = np.nanmin(xgb_eval_loss)
                    
        xgb_reg = xgb.XGBRegressor(n_estimators=xgb_best_nestimator, 
                                    tree_method='exact',
                                    learning_rate=xgb_best_lr,
                                    reg_lambda=xgb_best_lambd)
        xgb_reg.fit(X_train, y_train)
        y_xgb_pred = xgb_reg.predict(X_test)
        xgb_r2 = r2_score(y_test, y_xgb_pred)
        xgb_loss = mean_squared_error(y_test, y_xgb_pred)
        xgb_best_r2_list.append(xgb_r2)
        xgb_best_nestimator_list.append(xgb_best_nestimator)
        xgb_best_loss_list.append(xgb_loss)
        
            
        # Saving...
        current_path = os.path.join(folder_path, dataset, f'shuffle_{seed}', 'xgb', 'evaluate')
        if not os.path.exists(current_path):
            os.makedirs(current_path)
        np.savez(os.path.join(current_path, "data.npz"), 
            loss = xgb_loss,
            r2 = xgb_r2,
            lr = xgb_best_lr,
            lambd = xgb_best_lambd,
            n_estimators = xgb_best_nestimator
        )

        # gbdt
        gbdt_best_nestimator = n_estimators
        gbdt_best_eval_loss = 100
        gbdt_best_lr = 0.1
        for lr in lr_list:
            gbdt_reg = GradientBoostingRegressor(n_estimators=n_estimators,
                                                    learning_rate=lr,)
            gbdt_reg.fit(train_x, train_y)
            gbdt_eval_loss = gbdt_reg_score(gbdt_reg, eval_x, eval_y, n_estimators)[0]
            if np.nanmin(gbdt_eval_loss) < gbdt_best_eval_loss:
                gbdt_best_nestimator = np.nanargmin(gbdt_eval_loss)+1
                gbdt_best_lr = lr
                gbdt_best_eval_loss = np.nanmin(gbdt_eval_loss)
                
        gbdt_reg = GradientBoostingRegressor(n_estimators=gbdt_best_nestimator,
                                            learning_rate=gbdt_best_lr,)
        gbdt_reg.fit(X_train, y_train)
        y_gbdt_pred = gbdt_reg.predict(X_test)
        gbdt_r2 = r2_score(y_test, y_gbdt_pred)
        gbdt_loss = mean_squared_error(y_test, y_gbdt_pred)
        gbdt_best_r2_list.append(gbdt_r2)
        gbdt_best_nestimator_list.append(gbdt_best_nestimator)
        gbdt_best_loss_list.append(gbdt_loss)
 
            
        # Saving...
        current_path = os.path.join(folder_path, dataset, f'shuffle_{seed}', 'gbdt', 'evaluate')

        if not os.path.exists(current_path):
            os.makedirs(current_path)

        np.savez(os.path.join(current_path, "data.npz"), 
            loss = gbdt_loss,
            r2 = gbdt_r2,
            lr = gbdt_best_lr,
            n_estimators = gbdt_best_nestimator,
        )
            
    np.savez(os.path.join(folder_path, dataset, 'metric_lists.npz'),
        trb_r2 = trb_best_r2_list,
        xgb_r2 = xgb_best_r2_list,
        gbdt_r2 = gbdt_best_r2_list,
        trb_nestimator = trb_best_nestimator_list,
        xgb_nestimator = xgb_best_nestimator_list,
        gbdt_nestimator = gbdt_best_nestimator_list,
        trb_loss = trb_best_loss_list,
        xgb_loss = xgb_best_loss_list,
        gbdt_loss = gbdt_best_loss_list,  
    )
    print('-'*10)
