from __future__ import annotations
import pandas as pd
import numpy as np
from itertools import combinations
import builtins
from dataclasses import dataclass
import random


def gini_impurity(y: pd.Series) -> float:
    proportions = y.value_counts(normalize=True)
    return 1 - np.sum(proportions**2)


def gini_gain(y: pd.Series, left_mask: pd.Series, right_mask: pd.Series) -> float:
    if left_mask.sum() == 0 or right_mask.sum() == 0:
        return -np.inf
    
    left_impurity = gini_impurity(y[left_mask])
    right_impurity = gini_impurity(y[right_mask])
    weighted_impurity = (
        left_mask.sum() * left_impurity + right_mask.sum() * right_impurity
    ) / len(y)

    return gini_impurity(y) - weighted_impurity


def get_splits_continuous(col: pd.Series, num_splits: int = 20):

    col = col.dropna()
    unique_vals = np.sort(col.unique())

    if len(unique_vals) <= 1:
        return  # nothing to split

    if len(unique_vals) <= num_splits:
        # Use all midpoints between adjacent values
        thresholds = (unique_vals[:-1] + unique_vals[1:]) / 2
    else:
        # Use quantile-based thresholds
        percentiles = np.linspace(0, 100, num_splits + 2)[1:-1]  # drop 0% and 100%
        thresholds = np.unique(np.percentile(col, percentiles))

    for t in thresholds:
        left_mask = col <= t
        yield float(t), left_mask, ~left_mask


def get_splits_categorical(col: pd.Series):
    unique_vals = col.dropna().unique()
    n = len(unique_vals)
    if n <= 1:
        return

    for i in range(1, n // 2 + 1):
        for subset in combinations(unique_vals, i):
            subset = set(subset)
            left_mask = col.isin(subset)
            yield {str(c) for c in subset}, left_mask, ~left_mask


def gini_best_splits(X: pd.DataFrame, y: pd.Series):
    best_splits = [(None, None, -np.inf)]

    for col in X.columns:
        x_col = X[col]
        best_gain = -np.inf
        best_condition = None

        if pd.api.types.is_numeric_dtype(x_col):
            splitter = get_splits_continuous(x_col)
        else:
            splitter = get_splits_categorical(x_col)

        for condition, left_mask, right_mask in splitter:
            gain = gini_gain(y, left_mask, right_mask)
            if gain > best_gain:
                best_gain = gain
                best_condition = condition

        if best_gain == best_splits[0][2]:
            best_splits.append((col, best_condition, best_gain))
        elif best_gain > best_splits[0][2]:
            best_splits = [(col, best_condition, best_gain)]

    return best_splits


def should_stop(y, depth, best_gain, min_samples_split, max_depth):
    if len(y) < min_samples_split:
        return True
    if gini_impurity(y) == 0:
        return True
    if best_gain <= 0:
        return True
    if depth >= max_depth:
        return True
    return False


@dataclass
class CartNode:
    left: CartNode | None
    right: CartNode | None
    split_column: str | None
    condition: float | set[str] | None
    probabilities: dict[str, float]

    def split(self, X: pd.DataFrame) -> tuple[tuple[CartNode, pd.DataFrame], tuple[CartNode, pd.DataFrame]]:
        match type(self.condition):
            case builtins.set:
                left_mask = X[self.split_column].isin(self.condition)
            case builtins.float:
                left_mask = X[self.split_column] <= self.condition
            case _:
                raise ValueError("Condition is not valid!")
        return (
            (self.left, left_mask),
            (self.right, ~left_mask)
        )


class Cart:

    def __init__(self, X_train: pd.DataFrame, y_train: pd.Series):
        self.X_train = X_train
        self.y_train = y_train
        self.root = CartNode(None, None, None, None, self.get_probs_as_dict(y_train))

    def get_probs_as_dict(self, y: pd.Series):
        return {
            str(v): sum(y == v)/len(y)
            for v in self.y_train.unique()
        }

    def predict(self, X: pd.DataFrame):
        assert list(X.columns) == list(self.X_train.columns), "Inference data does not have the same columns as train data"
        X.index = range(len(X))
        y_cols = [str(c) for c in self.y_train.unique()]
        nodes_to_visit = [(self.root, X)]

        predictions = pd.DataFrame(columns=y_cols)

        while len(nodes_to_visit) > 0:
            node, X_temp = nodes_to_visit.pop()
            assert node is not None, "Arrived at a non-existing node"

            if not node.split_column:
                predictions = pd.concat([
                    predictions,
                    pd.DataFrame(
                        index=X_temp.index,
                        data=[{v: node.probabilities.get(str(v), 0) for v in y_cols}]
                    )
                ])
            else:
                (left_node, left_mask), (right_node, right_mask) = node.split(X_temp)
                nodes_to_visit.extend([
                    (left_node, X[left_mask]),
                    (right_node, X[right_mask])
                ])
        
        return predictions.sort_index()

    def fit(self):
        best_splits = gini_best_splits(self.X_train, self.y_train)

        chosen_split = best_splits[random.randint(0, len(best_splits)-1)]
        
        self.root.split_column = chosen_split[0]
        self.root.condition = chosen_split[1]

        (_, left_mask), (_, right_mask) = self.root.split(self.X_train)
        left_y = self.y_train[left_mask]
        self.root.left = CartNode(None, None, None, None, 
                                  self.get_probs_as_dict(left_y))
        right_y = self.y_train[right_mask]
        self.root.right = CartNode(None, None, None, None, 
                                   self.get_probs_as_dict(right_y))
