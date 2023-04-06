import numpy as np
from trb.node import Node
from trb.split_finder import SplitFinderVec


class Tree:
    def __init__(self,
                 objective,
                 max_depth,
                 epsilon,
                 min_child_weight,
                 min_leaf,
                 alpha,
                 beta):  
        self._objective = objective
        self._max_depth = max_depth
        self._root = None
        self._epsilon = epsilon
        self._min_child_weight = min_child_weight
        self._min_leaf = min_leaf
        self._alpha = alpha
        self._beta = beta

    def split(self, instances, labels, last_predictions):
        gradients = self._objective.gradients(labels, last_predictions)
        hessians = self._objective.hessians(labels, last_predictions)

        split_finder = SplitFinderVec(epsilon=self._epsilon,
                                      min_child_weight=self._min_child_weight,
                                      alpha=self._alpha,
                                      beta=self._beta)

        self._root = Node(self._max_depth, 
                          self._min_child_weight, self._min_leaf)
        self._root.split(split_finder, instances, gradients, hessians, depth=0)

    def predict(self, instance):
        return self._root.predict(instance)

    def get_dump(self):
        return '\n'.join(self._get_dump(self._root, depth=0, end=[]))

    @staticmethod
    def _vertical_lines(end):
        vertical_lines = []
        for e in np.roll(end, 1)[1:]:
            if e:
                vertical_lines.append('    ')
            else:
                vertical_lines.append('\u2502' + ' ' * 3)
        return ''.join(vertical_lines)

    @staticmethod
    def _horizontal_line(last_node):
        if last_node:
            return '\u2514\u2500\u2500'
        else:
            return '\u251c\u2500\u2500'

    def _get_dump(self, node, depth, end, index=0):
        dump = []
        if depth > 0:
            indent = self._vertical_lines(end) + self._horizontal_line(end[-1])
            dump.append(f'{indent} {node}')
        else:
            dump.append(f'{node}')
        if node.left_child is not None:
            dump.extend(self._get_dump(node.left_child, depth + 1, end + [False]))
        if node.right_child is not None:
            dump.extend(self._get_dump(node.right_child, depth + 1, end + [True]))
        return dump

     