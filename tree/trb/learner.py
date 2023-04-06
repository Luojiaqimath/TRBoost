import numpy as np
import math
from trb.tree import Tree


class TRBLearner:

    def __init__(self,
                 objective,
                 n_estimators,
                 alpha=0.1,
                 beta=10,
                 max_depth=6,
                 base_score=0.5,
                 min_child_weight=0.1,
                 min_leaf=2,
                 rho1=0.9,
                 rho2=1.1,
                 gamma=1.01,
                 eta=0.01,
                 update_strategy='standard',
                 ):
        self._objective = objective
        self._n_estimators = n_estimators  # boosting number
        self._base_score = base_score  # f_0
        self._max_depth = max_depth  

        self._tree = None
        self._trees = []  # set of boosting tree
        self._epsilon = 1e-6  
        self._min_child_weight = min_child_weight
        self._min_leaf = min_leaf  # Minimum number of samples in a node

        self._alpha = alpha  # radius1
        self._beta = beta  # radius2
        self._rho_1 = rho1  # judgement condition1 in trust region
        self._rho_2 = rho2  # judgement condition2 in trust region
        self._gamma = gamma  
        self._eta = eta  
        self._update_strategy = update_strategy

        self.train_loss_list = []
        self.eval_loss_list = []
        self.train_metric_list = []
        self.eval_metric_list = []
        self.rho_list = []

    def fit(self, train_set, eval_set=None):
        instances_train = train_set[0]
        labels_train = train_set[1]
        predictions = np.full(len(labels_train), self._base_score)  # y_0

        initial_loss = self._objective.loss(
            labels_train, predictions)  # loss_0
        for i in range(self._n_estimators):
            # print('tree {} '.format(i+1)+'-'*10)

            # initialization
            prev_loss = self._objective.loss(
                labels_train, predictions)  # loss_(k-1)
            prev_pred = predictions.copy()  # y_(k-1)

            tree = Tree(self._objective,
                        max_depth=self._max_depth,
                        epsilon=self._epsilon,
                        min_child_weight=self._min_child_weight,
                        min_leaf=self._min_leaf,
                        alpha=self._alpha,
                        beta=self._beta,
                        min_nodenum=self._min_nodenum
                        )  # f_k

            tree.split(instances_train, labels_train,
                       last_predictions=prev_pred)
            self._trees.append(tree)
            predictions = self.predict(instances_train)  # y_k
            train_loss = self._objective.loss(
                labels_train, predictions)  # loss_k
            self.train_loss_list.append(train_loss)
            train_metric = self._objective.metric(labels_train, predictions)
            self.train_metric_list.append(train_metric)

            pred_k = self._objective.inverse_trans(predictions) -\
                self._objective.inverse_trans(prev_pred)  # f_k(x)

            grad = self._objective.gradients(labels_train, prev_pred)  # g_k-1
            hess = self._objective.hessians(labels_train, prev_pred)  # h_k-1

            # calculate rho_k
            if self._update_strategy == 'standard':
                mk_residual = np.multiply(
                    grad, pred_k) + 0.5 * np.multiply(hess, pred_k**2)
                rho_k = np.divide(prev_loss-train_loss, -np.mean(mk_residual))

            elif self._update_strategy == 'gradient':
                rho_k = np.divide(prev_loss-train_loss,
                                  np.mean(np.abs(pred_k)))

            else:
                rho_k = np.divide(prev_loss-train_loss,
                                  initial_loss-train_loss)

            self.rho_list.append(rho_k)

            if self._rho_1 < rho_k < self._rho_2:
                # print('keep mu')
                pass
            else:
                self._alpha = self._alpha*self._gamma
                self._beta = self._beta*self._gamma
                # print(f'enlarge alpha and beta:{self._alpha, self._beta}')

            if self._eta < rho_k:
                if eval_set is None:
                    # print(f'train_loss:{train_loss:.4f}')
                    pass
                else:
                    instances_eval = eval_set[0]
                    labels_eval = eval_set[1]
                    pred_eval = self.predict(instances_eval)
                    eval_loss = self._objective.loss(labels_eval, pred_eval)
                    self.eval_loss_list.append(eval_loss)
                    eval_metric = self._objective.metric(
                        labels_eval, pred_eval)
                    self.eval_metric_list.append(eval_metric)
            else:
                self._trees.pop(-1)
                predictions = prev_pred
                if eval_set is not None:
                    if len(self._trees) != 0:
                        self.eval_loss_list.append(eval_loss)  # eval loss in the previous step
                        self.eval_metric_list.append(
                            eval_metric)  # eval metric in the previous step
                    else:
                        self.eval_loss_list.append(np.nan)
                        self.eval_metric_list.append(np.nan)
        return self

    def predict(self, instances):
        predictions = np.full(len(instances), self._base_score)
        pred_k = np.full(len(instances), 0)
        for i, tree in enumerate(self._trees):
            for index, instance in enumerate(instances):
                prediction = tree.predict(instance)
                predictions[index] += prediction
                if i == len(self._trees)-1:
                    pred_k[index] += prediction
        return self._objective.transformation(predictions)

    def score(self, X, y):
        y_pred, _ = self.predict(X)
        return self._objective.loss(y, y_pred)

    def get_params(self, deep=True):
        return {'objective': self._objective,
                'n_estimators': self._n_estimators,
                'base_score': self._base_score,
                'max_depth': self._max_depth,
                'min_child_weight': self._min_child_weight,
                'min_leaf': self._min_leaf,
                'alpha': self._alpha,
                'beta': self._beta,
                'rho1': self._rho_1,
                'rho2': self._rho_2,
                'gamma': self._gamma,
                'eta': self._eta,
                'update_strategy': self._update_strategy}
