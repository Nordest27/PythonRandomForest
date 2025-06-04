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
    
    for val in unique_vals:
        left_mask = col == val
        yield str(val), left_mask, ~left_mask


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


@dataclass
class CartNode:
    left: CartNode | None
    right: CartNode | None
    split_column: str | None
    condition: float | str | None
    probabilities: dict[str, float]

    def split(self, X: pd.DataFrame) -> tuple[tuple[CartNode, pd.Series], tuple[CartNode, pd.Series]]:
        match type(self.condition):
            case builtins.str | builtins.int:
                left_mask = X[self.split_column].astype(str) == self.condition
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
                    (left_node, X_temp[left_mask]),
                    (right_node, X_temp[right_mask])
                ])
        
        return predictions.sort_index()

    def fit(self, min_samples_split=5, max_depth=10):

        def should_stop(y, best_gain, depth):
            return (
                len(y) < min_samples_split or
                gini_impurity(y) == 0 or
                best_gain <= 0 or
                depth >= max_depth
            )
        
        def expand_tree(node: CartNode, mask: pd.Series, depth: int):
            X_temp = self.X_train[mask]
            y_temp = self.y_train[mask]

            best_splits = gini_best_splits(X_temp, y_temp)

            if should_stop(y_temp, best_splits[0][2], depth):
                return

            chosen_split = best_splits[random.randint(0, len(best_splits)-1)]
            
            node.split_column = chosen_split[0]
            node.condition = chosen_split[1]

            (_, left_mask), (_, right_mask) = node.split(self.X_train)
            left_mask = left_mask & mask
            right_mask = right_mask & mask

            left_y = self.y_train[left_mask]
            node.left = CartNode(None, None, None, None, 
                                    self.get_probs_as_dict(left_y))
            expand_tree(node.left, left_mask, depth+1)

            right_y = self.y_train[right_mask]
            node.right = CartNode(None, None, None, None, 
                                    self.get_probs_as_dict(right_y))
            expand_tree(node.right, right_mask, depth+1)
        
        expand_tree(self.root, self.y_train==self.y_train, 0)

    def print_tree(self):
        def print_node(node: CartNode, prefix: str = "", is_left: bool = True):
            connector = "├── " if is_left else "└── "
            if node.split_column is None:
                probs = ", ".join(f"{k}: {v:.2f}" for k, v in node.probabilities.items())
                print(f"{prefix}{connector}Leaf: [{probs}]")
            else:
                if isinstance(node.condition, str):
                    cond_str = f"is {node.condition}"
                else:
                    cond_str = f"<= {node.condition:.3f}"
                print(f"{prefix}{connector}{node.split_column} {cond_str}")
                child_prefix = prefix + ("│   " if is_left else "    ")
                print_node(node.left, child_prefix, True)
                print_node(node.right, child_prefix, False)

        print_node(self.root, "", False)
