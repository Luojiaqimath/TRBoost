import numpy as np


class SplitFinder:

    def __init__(self, epsilon, min_child_weight, alpha, beta):
        self._epsilon = epsilon
        self._min_child_weight = min_child_weight
        self._alpha = alpha
        self._beta = beta

    def calc_weight(self, gradients, hessians):
        sum_gradients = np.sum(gradients)
        sum_hessians = np.sum(hessians)
        num_instance = gradients.shape[0]
        mu_weight = self._alpha*num_instance+self._beta
        return -sum_gradients/(sum_hessians+mu_weight)


class SplitFinderVec(SplitFinder):

    def _calc_gain(self, gradient_sum, hessian_sum, mu_sum):

        gradient_sum = gradient_sum.reshape(-1)
        hessian_sum = hessian_sum.reshape(-1)
        weight = -gradient_sum/(hessian_sum+mu_sum)
        return gradient_sum * weight + 0.5 * hessian_sum * weight**2

    def find_split(self, instances, gradients, hessians):
        num_instances = len(instances)
        num_features = len(instances[0])
        gradient_sum = gradients.sum()
        hessian_sum = hessians.sum()
        root_mu = self._alpha*num_instances+self._beta
        root_gain = self._calc_gain(gradient_sum, hessian_sum, root_mu)

        # Sort instances in a descending order
        sorted_instance_indices = instances.argsort(axis=0)
        sorted_instance_indices = np.flip(sorted_instance_indices, axis=0)
        instances_flatten = instances.flatten(order='F')

        # Compute feature values and last feature values
        tmp = sorted_instance_indices + \
            (np.arange(num_features) * num_instances)
        feature_values = instances_flatten[tmp[1:].flatten(order='F')]
        last_feature_values = instances_flatten[tmp[:-1].flatten(order='F')]

        # Greater equal
        gradient_sum_ge = np.cumsum(gradients[sorted_instance_indices[:-1]], axis=0).flatten(
            order='F')
        hessian_sum_ge = np.cumsum(hessians[sorted_instance_indices[:-1]], axis=0).flatten(
            order='F')
        mu_ge = self._alpha*np.tile(np.arange(1, len(sorted_instance_indices)),
                                    num_features) + self._beta

        # Lower than
        gradient_sum_lt = gradient_sum - gradient_sum_ge
        hessian_sum_lt = hessian_sum - hessian_sum_ge
        mu_lt = self._alpha*np.tile(np.arange(1, len(sorted_instance_indices))[::-1],
                                    num_features) + self._beta

        loss_changes = - self._calc_gain(gradient_sum_ge, hessian_sum_ge, mu_ge) - \
            self._calc_gain(gradient_sum_lt, hessian_sum_lt, mu_lt) + \
            root_gain

        # Consider loss change only when hessian_sum_lt_new >= min_child_weight
        hessian_sum_lt_new = np.array(
            [h if h > 0 else h+self._alpha for h in hessian_sum_lt])
        loss_changes = np.where(np.less(hessian_sum_lt_new, self._min_child_weight), self._epsilon,
                                loss_changes)

        # Consider loss change only when hessian_sum_gt+alpha*n >= min_child_weight
        hessian_sum_ge_new = np.array(
            [h if h > 0 else h+self._alpha for h in hessian_sum_ge])
        loss_changes = np.where(np.less(hessian_sum_ge_new, self._min_child_weight), self._epsilon,
                                loss_changes)

        # Consider loss change only when feature_value != last_feature_value
        loss_changes = np.where(np.invert(np.not_equal(feature_values, last_feature_values)),
                                self._epsilon,
                                loss_changes)

        if len(loss_changes) == 0:
            return None, None, None, None

        best_loss_change_index = np.argmax(loss_changes)
        best_feature_index, best_split_index = np.unravel_index(best_loss_change_index, (
            num_features, num_instances - 1))

        best_loss_change = loss_changes[best_loss_change_index]
        if last_feature_values[best_split_index] is None:
            best_split_value = None
        else:
            best_split_value = (feature_values[best_loss_change_index] + last_feature_values[
                best_loss_change_index]) * 0.5

        if best_loss_change <= self._epsilon:
            # If loss change is smaller or equal to 0.0 then no split is optimal
            # Epsilon is used instead of 0.0 because of floating point errors
            return None, None, None, None

        # Select only the indices of the best feature
        sorted_indices_best_feature = sorted_instance_indices[:,
                                                              best_feature_index]

        # Index of the instances where value >= split_value
        best_indices_lt = sorted_indices_best_feature[best_split_index + 1:]

        # Index of the instances where value < split value
        best_indices_ge = sorted_indices_best_feature[:best_split_index + 1]

        return best_feature_index, best_split_value, best_indices_lt, best_indices_ge
