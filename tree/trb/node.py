import numpy as np


class Node:
    def __init__(self, max_depth, min_child_weight, min_leaf):
        self.left_child = None
        self.right_child = None
        self.split_value = None
        self.feature_index = None
        self.is_leaf = None
        self._max_depth = max_depth
        self._min_child_weight = min_child_weight
        self._min_leaf = min_leaf

    def __str__(self):
        if self.is_leaf:
            return f'leaf={self.weight}'
        else:
            return f'[feature_{self.feature_index}_value<{self.split_value}]'

    def split(self, split_finder, instances, gradients, hessians, depth):

        sum_hessians = np.sum(hessians)
        if ((gradients.shape[0] < self._min_leaf) or
            (depth == self._max_depth) or
                (sum_hessians != 0 and np.abs(sum_hessians) < self._min_child_weight)):

            self.is_leaf = True
            self.weight = split_finder.calc_weight(gradients, hessians)
            return

        feature_index, split_value, best_indexes_lt, best_indexes_ge = \
            split_finder.find_split(instances, gradients, hessians)

        if split_value is None:
            self.is_leaf = True
            self.weight = split_finder.calc_weight(gradients, hessians)
            return
        else:
            # Init new node
            self.split_value = split_value
            self.feature_index = feature_index
            self.left_child = Node(self._max_depth,
                                   self._min_child_weight, self._min_leaf)
            self.right_child = Node(self._max_depth,
                                    self._min_child_weight, self._min_leaf)

        # value < split_value
        self.left_child.split(
            split_finder,
            instances[best_indexes_lt], gradients[best_indexes_lt], hessians[best_indexes_lt],
            depth=depth + 1,
        )

        # value >= split_value
        self.right_child.split(
            split_finder,
            instances[best_indexes_ge], gradients[best_indexes_ge], hessians[best_indexes_ge],
            depth=depth + 1)

    def predict(self, instance):
        if self.is_leaf:
            return self.weight
        elif instance[self.feature_index] < self.split_value:
            if self.left_child:
                return self.left_child.predict(instance)
        else:
            if self.right_child:
                return self.right_child.predict(instance)
