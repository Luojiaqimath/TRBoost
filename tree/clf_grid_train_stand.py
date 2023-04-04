import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, log_loss
from sklearn import datasets
import matplotlib.pyplot as plt
from trb import objectives
from trb.learner import TRBLearner
import xgboost as xgb
from sklearn.ensemble import GradientBoostingClassifier
import data_clf
import pickle
import warnings
warnings.filterwarnings('ignore')


def gbdt_clf_score(clf, X_test, y_test, n_estimators):
    loss = np.zeros((n_estimators,), dtype=np.float64)
    auc = np.zeros((n_estimators,), dtype=np.float64)
    f1 = np.zeros((n_estimators,), dtype=np.float64)
    for i, y_prob in enumerate(clf.staged_predict_proba(X_test)):
        loss[i] = log_loss(y_test, y_prob[:, 1])
        auc[i] = roc_auc_score(y_test, y_prob[:, 1])
    for i, y_pred in enumerate(clf.staged_predict(X_test)):
        f1[i] = f1_score(y_test, y_pred)
    return loss, auc, f1


folder_path = "./clf_standard_result"
if not os.path.exists(folder_path):
    os.makedirs(folder_path)


objective = objectives.BinaryCrossentropy()
n_estimators = 100

for dataset in ['sonar', 'german', 'spam', 'credit', 'adult', 'electricity']:
    print('data: ' + dataset)
    X, y = data_clf.loader(dataset)

    trb_best_auc_list = []
    trb_best_f1_list = []
    trb_best_nestimator_list = []

    xgb_best_auc_list = []
    xgb_best_f1_list = []
    xgb_best_nestimator_list = []

    gbdt_best_auc_list = []
    gbdt_best_f1_list = []
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
                trb_clf = TRBLearner(objective,
                                     base_score=0.5,  # 回归为0
                                     n_estimators=n_estimators,
                                     alpha=_alpha,
                                     eta=_eta)
                trb_clf.fit(train_set=(train_x, train_y),
                            eval_set=(eval_x, eval_y))
                trb_eval_loss = trb_clf.eval_loss_list

                if np.nanmin(trb_eval_loss) < trb_best_eval_loss:
                    trb_best_nestimator = np.nanargmin(trb_eval_loss)+1
                    trb_best_alpha = _alpha
                    trb_best_eta = _eta
                    trb_best_eval_loss = np.nanmin(trb_eval_loss)

        trb_clf = TRBLearner(objective,
                             base_score=0.5,
                             n_estimators=trb_best_nestimator,
                             alpha=trb_best_alpha,
                             eta=trb_best_eta)
        trb_clf.fit(train_set=(X_train, y_train))
        y_trb_prob = trb_clf.predict(X_test)
        y_trb_cate = (y_trb_prob > 0.5)+0
        trb_auc = roc_auc_score(y_test, y_trb_prob)
        trb_f1 = f1_score(y_test, y_trb_cate)
        trb_best_auc_list.append(trb_auc)
        trb_best_f1_list.append(trb_f1)
        trb_best_nestimator_list.append(trb_best_nestimator)

        # Saving...
        current_path = os.path.join(
            folder_path, dataset, f'shuffle_{seed}', 'trb', 'evaluate')

        if not os.path.exists(current_path):
            os.makedirs(current_path)

        np.savez(os.path.join(current_path, "data.npz"),
                 auc=trb_auc,
                 f1=trb_f1,
                 alpha=trb_best_alpha,
                 eta=trb_best_eta,
                 n_estiamtors=trb_best_nestimator)

        # xgb
        lr_list = [0.1, 0.5, 1]
        lambd_list = [0, 0.5, 1, 5]
        xgb_best_nestimator = n_estimators
        xgb_best_eval_loss = 100
        xgb_best_lr = 0.1
        xgb_best_lambd = 0
        search_idx = 0
        for lr in lr_list:
            for lambd in lambd_list:
                xgb_clf = xgb.XGBClassifier(n_estimators=n_estimators,
                                            tree_method='exact',
                                            learning_rate=lr,
                                            reg_lambda=lambd)
                xgb_clf.fit(train_x, train_y)
                xgb_train_loss = []
                xgb_train_metric = []
                xgb_eval_loss = []
                xgb_eval_metric = []
                for i in range(n_estimators):
                    y_train_pred = xgb_clf.predict_proba(
                        train_x, iteration_range=(0, i+1))[:, 1]
                    y_eval_pred = xgb_clf.predict_proba(
                        eval_x, iteration_range=(0, i+1))[:, 1]
                    xgb_train_loss.append(log_loss(train_y, y_train_pred))
                    xgb_eval_loss.append(log_loss(eval_y, y_eval_pred))
                    xgb_train_metric.append(
                        roc_auc_score(train_y, y_train_pred))
                    xgb_eval_metric.append(roc_auc_score(eval_y, y_eval_pred))

                if np.nanmin(xgb_eval_loss) < xgb_best_eval_loss:
                    xgb_best_nestimator = np.nanargmin(xgb_eval_loss)+1
                    xgb_best_lr = lr
                    xgb_best_lambd = lambd
                    xgb_best_eval_loss = np.nanmin(xgb_eval_loss)

        xgb_clf = xgb.XGBClassifier(n_estimators=xgb_best_nestimator,
                                    tree_method='exact',
                                    learning_rate=xgb_best_lr,
                                    reg_lambda=xgb_best_lambd)
        xgb_clf.fit(X_train, y_train)
        y_xgb_prob = xgb_clf.predict_proba(X_test)[:, 1]
        y_xgb_cate = xgb_clf.predict(X_test)
        xgb_auc = roc_auc_score(y_test, y_xgb_prob)
        xgb_f1 = f1_score(y_test, y_xgb_cate)
        xgb_best_auc_list.append(xgb_auc)
        xgb_best_f1_list.append(xgb_f1)
        xgb_best_nestimator_list.append(xgb_best_nestimator)

        # Saving...
        current_path = os.path.join(
            folder_path, dataset, f'shuffle_{seed}', 'xgb', 'evaluate')

        if not os.path.exists(current_path):
            os.makedirs(current_path)

        np.savez(os.path.join(current_path, "data.npz"),
                 auc=xgb_auc,
                 f1=xgb_f1,
                 lr=xgb_best_lr,
                 lambd=xgb_best_lambd,
                 n_estimators=xgb_best_nestimator)

        # gbdt
        gbdt_best_nestimator = n_estimators
        gbdt_best_eval_loss = 100
        gbdt_best_lr = 0.1
        for lr in lr_list:
            gbdt_clf = GradientBoostingClassifier(n_estimators=n_estimators,
                                                  learning_rate=lr,)
            gbdt_clf.fit(train_x, train_y)
            gbdt_eval_loss = gbdt_clf_score(
                gbdt_clf, eval_x, eval_y, n_estimators)[0]

            if np.nanmin(gbdt_eval_loss) < gbdt_best_eval_loss:
                gbdt_best_nestimator = np.nanargmin(gbdt_eval_loss)+1
                gbdt_best_lr = lr
                gbdt_best_eval_loss = np.nanmin(gbdt_eval_loss)

        gbdt_clf = GradientBoostingClassifier(n_estimators=gbdt_best_nestimator,
                                              learning_rate=gbdt_best_lr,)
        gbdt_clf.fit(X_train, y_train)
        y_gbdt_prob = gbdt_clf.predict_proba(X_test)[:, 1]
        y_gbdt_cate = gbdt_clf.predict(X_test)
        gbdt_auc = roc_auc_score(y_test, y_gbdt_prob)
        gbdt_f1 = f1_score(y_test, y_gbdt_cate)
        gbdt_best_auc_list.append(gbdt_auc)
        gbdt_best_f1_list.append(gbdt_f1)
        gbdt_best_nestimator_list.append(gbdt_best_nestimator)

        # Saving...
        current_path = os.path.join(
            folder_path, dataset, f'shuffle_{seed}', 'gbdt', 'evaluate')
        if not os.path.exists(current_path):
            os.makedirs(current_path)
        np.savez(os.path.join(current_path, "data.npz"),
                 auc=gbdt_auc,
                 f1=gbdt_f1,
                 lr=gbdt_best_lr,
                 n_estimators=gbdt_best_nestimator,
                 )

    np.savez(os.path.join(folder_path, dataset, 'metric_lists.npz'),
             trb_auc=trb_best_auc_list,
             trb_f1=trb_best_f1_list,
             trb_nestimator=trb_best_nestimator_list,
             xgb_auc=xgb_best_auc_list,
             xgb_f1=xgb_best_f1_list,
             xgb_nestimator=xgb_best_nestimator_list,
             gbdt_auc=gbdt_best_auc_list,
             gbdt_f1=gbdt_best_f1_list,
             gbdt_nestimator=gbdt_best_nestimator_list,
             )

    print('-'*10)
