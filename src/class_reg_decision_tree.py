from __future__ import annotations
import pandas as pd
import numpy as np
from itertools import combinations
import builtins
from dataclasses import dataclass
import random
from tqdm import tqdm


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


def get_splits_continuous(col: pd.Series, num_splits: int = 5):
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


def get_random_splits_categorical(
        col: pd.Series, y: pd.Series, num_random: int = 5
    ):
    unique_vals = list(col.dropna().unique())
    n = len(unique_vals)

    if n <= 1:
        return

    # Try num_random random non-trivial subsets
    for _ in range(num_random):
        k = random.randint(1, n - 1)  # avoid empty or full set
        subset = set(random.sample(unique_vals, k))
        left_mask = col.isin(subset)
        right_mask = ~left_mask

        if left_mask.sum() == 0 or right_mask.sum() == 0:
            continue

        yield subset, left_mask, right_mask


def gini_best_splits(X: pd.DataFrame, y: pd.Series, features_to_explore: list | None = None):
    best_splits = [(None, None, -np.inf)]
    if features_to_explore is not None:
        features_to_explore = X.columns
    for col in features_to_explore:
        x_col = X[col]
        best_gain = -np.inf
        best_condition = None
        
        # 1. Handle regular splits
        if pd.api.types.is_numeric_dtype(x_col):
            splitter = get_splits_continuous(x_col)
        else:
            splitter = get_random_splits_categorical(x_col, y)

        for condition, left_mask, right_mask in splitter:
            gain = gini_gain(y, left_mask, right_mask)
            if gain > best_gain:
                best_gain = gain
                best_condition = condition

        # 2. Handle null split explicitly
        null_mask = x_col.isnull()
        non_null_mask = ~null_mask
        if null_mask.any() and non_null_mask.any():
            null_gain = gini_gain(y, null_mask, non_null_mask)
            if null_gain > best_gain:
                best_gain = null_gain
                best_condition = "__NULL__"

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
    condition: float | set | str | None
    probabilities: dict[str, float]

    def split(self, X: pd.DataFrame) -> tuple[tuple[CartNode, pd.Series], tuple[CartNode, pd.Series]]:
        match type(self.condition):
            case builtins.set:
                left_mask = X[self.split_column].isin(self.condition)
            case builtins.float:
                left_mask = X[self.split_column] <= self.condition
            case builtins.str:
                left_mask = X[self.split_column].isnull()
            case _:
                raise ValueError("Condition is not valid!")
        return (
            (self.left, left_mask),
            (self.right, ~left_mask)
        )


class RandCart:

    def __init__(
            self, 
            X_train: pd.DataFrame, 
            y_train: pd.Series, 
            max_features_to_explore: int = 5,
            max_depth: int = 10,
            use_progress_bar: bool = False
        ):
        self.X_train = X_train
        self.y_train = y_train
        self.max_features_to_explore = min(max_features_to_explore, len(X_train.columns)//2)
        self.max_depth = max_depth
        self.root = CartNode(None, None, None, None, self.get_probs_as_dict(y_train))
        self.use_progress_bar = use_progress_bar

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

    def fit(self, min_samples_split=5):
        if self.use_progress_bar:
            max_nodes = 2 ** (self.max_depth + 1) - 1  # Conservative full tree estimate
            pbar = tqdm(total=max_nodes, desc="Training RandCart")

        def should_stop(y, best_gain, depth):
            return (
                len(y) < min_samples_split or
                gini_impurity(y) == 0 or
                best_gain <= 0 or
                depth >= self.max_depth
            )
        
        def expand_tree(node: CartNode, index: pd.Index, depth: int):
            X_temp = self.X_train.loc[index]
            random_features = np.random.choice(X_temp.columns, self.max_features_to_explore)

            y_temp = self.y_train.loc[index]

            best_splits = gini_best_splits(X_temp, y_temp, random_features)

            if should_stop(y_temp, best_splits[0][2], depth):
                if self.use_progress_bar:
                    pbar.update(2 ** (self.max_depth + 1 - depth) - 1)
                return

            chosen_split = best_splits[random.randint(0, len(best_splits)-1)]
            if self.use_progress_bar:
                pbar.update(1)

            node.split_column = chosen_split[0]
            node.condition = chosen_split[1]

            (_, left_mask), (_, right_mask) = node.split(X_temp)

            left_y = y_temp[left_mask]
            node.left = CartNode(None, None, None, None, self.get_probs_as_dict(left_y))
            expand_tree(node.left, left_y.index, depth+1)

            right_y = y_temp[right_mask]
            node.right = CartNode(None, None, None, None, self.get_probs_as_dict(right_y))
            expand_tree(node.right, right_y.index, depth+1)


        expand_tree(self.root, self.y_train.index, 0)
        pbar.close()

    def print_tree(self):
        def print_node(node: CartNode, prefix: str = "", is_left: bool = True):
            connector = "├── " if is_left else "└── "
            if node.split_column is None:
                probs = ", ".join(f"{k}: {v:.2f}" for k, v in node.probabilities.items())
                print(f"{prefix}{connector}Leaf: [{probs}]")
            else:
                if isinstance(node.condition, set):
                    cond_str = f"is in {node.condition}"
                elif isinstance(node.condition, str):
                    cond_str = "is null"
                else:
                    cond_str = f"<= {node.condition:.3f}"
                print(f"{prefix}{connector}{node.split_column} {cond_str}")
                child_prefix = prefix + ("│   " if is_left else "    ")
                print_node(node.left, child_prefix, True)
                print_node(node.right, child_prefix, False)

        print_node(self.root, "", False)
