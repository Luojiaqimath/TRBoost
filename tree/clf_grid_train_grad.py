import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss, f1_score
import matplotlib.pyplot as plt
import data_clf
import pickle
from trb import objectives
from trb.learner import TRBLearner
import warnings
warnings.filterwarnings('ignore')


folder_path = "./clf_gradient_result"
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
        search_idx = 0

        for _alpha in alpha_list:
            for _eta in eta_list:
                trb_clf = TRBLearner(objective,
                                     base_score=0.5,
                                     n_estimators=n_estimators,
                                     alpha=_alpha,
                                     eta=_eta,
                                     update_strategy='gradient')
                trb_clf.fit(train_set=(train_x, train_y),
                            eval_set=(eval_x, eval_y))
                trb_eval_loss = trb_clf.eval_loss_list

                if np.nanmin(trb_eval_loss) < trb_best_eval_loss:
                    trb_best_nestimator = np.nanargmin(trb_eval_loss)+1
                    trb_best_alpha = _alpha
                    trb_best_eta = _eta
                    trb_best_eval_loss = np.nanmin(trb_eval_loss)

                search_idx += 1

        trb_clf = TRBLearner(objective,
                             base_score=0.5,
                             n_estimators=trb_best_nestimator,
                             alpha=trb_best_alpha,
                             eta=trb_best_eta,
                             update_strategy='gradient')
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
                 n_estiamtors=trb_best_nestimator,
                 )

    np.savez(os.path.join(folder_path, dataset, 'metric_lists.npz'),
             trb_auc=trb_best_auc_list,
             trb_f1=trb_best_f1_list,
             trb_nestimator=trb_best_nestimator_list,
             )
    print('-'*10)
