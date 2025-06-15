from __future__ import annotations
import pandas as pd
import numpy as np
from dataclasses import dataclass
import random
from tqdm import tqdm
import warnings

# Ignore: tqdmWarning: clamping frac to range [0, 1]
warnings.filterwarnings("ignore")


def gini_impurity(y: np.ndarray) -> float:
    """Computes the Gini impurity of the label vector y.
    It measures the probability of misclassification by
    randomly picking a label according to the distribution
    of classes in y"""
    if len(y) == 0:
        return 0
    _, counts = np.unique(y, return_counts=True)
    proportions = counts / len(y)
    return 1 - np.sum(proportions**2)


def gini_gain(y: np.ndarray, left_mask: np.ndarray, right_mask: np.ndarray) -> float:
    """Computes the Gini gain from splitting y into two parts ,
    using left_mask and right_mask to define the split.
    It represents the reduction in impurity after the split"""
    left_count = np.sum(left_mask)
    right_count = np.sum(right_mask)

    if left_count == 0 or right_count == 0:
        return -np.inf

    total_count = len(y)
    left_impurity = gini_impurity(y[left_mask])
    right_impurity = gini_impurity(y[right_mask])

    weighted_impurity = (
        left_count * left_impurity + right_count * right_impurity
    ) / total_count
    return gini_impurity(y) - weighted_impurity


def get_splits_continuous(col: np.ndarray, num_splits: int = 20):
    """Returns a list of num_splits threshold values to try
    for splitting a continuous column. These are sampled
    from the unique values in cola"""
    unique_vals = np.unique(col[~np.isnan(col)])  # Remove NaNs first

    if len(unique_vals) <= 1:
        return

    if len(unique_vals) <= num_splits:
        thresholds = (unique_vals[:-1] + unique_vals[1:]) / 2
    else:
        percentiles = np.linspace(0, 100, num_splits + 2)[1:-1]
        thresholds = np.unique(np.percentile(unique_vals, percentiles))

    for t in thresholds:
        left_mask = col <= t
        yield float(t), left_mask, ~left_mask


def get_random_splits_categorical(col: np.ndarray, num_random: int = 20):
    """Returns num_random randomly generated binary splits
    of the values in a categorical column. Each split is
    represented as a set of categories ."""
    # Remove NaNs and get unique values
    mask = ~pd.isna(col)
    if not mask.any():
        return

    unique_vals = np.unique(col[mask])
    n = len(unique_vals)

    if n <= 1:
        return

    for _ in range(num_random):
        k = random.randint(1, n - 1)
        subset = set(np.random.choice(unique_vals, k, replace=False))
        left_mask = np.isin(col, list(subset))
        right_mask = ~left_mask

        if not left_mask.any() or not right_mask.any():
            continue

        yield subset, left_mask, right_mask


def gini_best_splits(
    X: np.ndarray,
    y: np.ndarray,
    dtypes: dict,
    feature_names: list,
    features_to_explore: list | None = None,
):
    """For a random subset of features (given by features_to_explore ),
    calculates candidate splits and their gini gains.
    Returns the best split found (feature , condition , gain )."""
    best_splits = [(None, None, -np.inf)]

    if features_to_explore is None:
        features_to_explore = list(range(len(feature_names)))
    else:
        # Convert feature names to indices
        features_to_explore = [
            feature_names.index(f) for f in features_to_explore if f in feature_names
        ]

    for col_idx in features_to_explore:
        x_col = X[:, col_idx]
        best_gain = -np.inf
        best_condition = None

        # Handle regular splits
        if np.issubdtype(dtypes[feature_names[col_idx]], np.number):
            splitter = get_splits_continuous(x_col.astype(float))
        else:
            splitter = get_random_splits_categorical(x_col)

        for condition, left_mask, right_mask in splitter:
            gain = gini_gain(y, left_mask, right_mask)
            if gain > best_gain:
                best_gain = gain
                best_condition = condition

        # Handle null split
        null_mask = pd.isna(x_col)
        non_null_mask = ~null_mask
        if null_mask.any() and non_null_mask.any():
            null_gain = gini_gain(y, null_mask, non_null_mask)
            if null_gain > best_gain:
                best_gain = null_gain
                best_condition = "__NULL__"

        if best_gain == best_splits[0][2]:
            best_splits.append((feature_names[col_idx], best_condition, best_gain))
        elif best_gain > best_splits[0][2]:
            best_splits = [(feature_names[col_idx], best_condition, best_gain)]

    return best_splits


@dataclass
class CartNode:
    left: CartNode | None
    right: CartNode | None
    split_column: str | None
    condition: float | set | str | None
    probabilities: dict[str, float]

    def split(self, X: np.ndarray, col_idx: int) -> tuple[np.ndarray, np.ndarray]:
        """Partitions input data X based on the Node ’s condition applied
        to the column col_idx , which is gotten from the split_column field"""
        x_col = X[:, col_idx]

        if isinstance(self.condition, set):
            left_mask = np.isin(x_col, list(self.condition))
        elif isinstance(self.condition, float):
            left_mask = x_col.astype(float) <= self.condition
        elif isinstance(self.condition, str):
            left_mask = pd.isna(x_col)
        else:
            raise ValueError("Condition is not valid!")

        return left_mask, ~left_mask


class RandCart:
    def __init__(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        max_features_to_explore: int = 5,
        max_depth: int = 10,
        min_samples_split: int = 5,
        use_progress_bar: bool = False,
    ):
        # Convert to numpy arrays for faster computation
        self.X_train_np = X_train.values
        self.y_train_np = y_train.values
        self.feature_names = list(X_train.columns)
        self.class_names = [str(c) for c in y_train.unique()]
        self.dtypes = {
            c: (dt if isinstance(dt, np.dtype) else np.dtypes.ObjectDType)
            for c, dt in X_train.dtypes.to_dict().items()
        }
        self.max_features_to_explore = min(
            max_features_to_explore, len(X_train.columns) // 2
        )
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = CartNode(
            None, None, None, None, self.get_probs_as_dict(y_train.values)
        )
        self.use_progress_bar = use_progress_bar

    def get_probs_as_dict(self, y: np.ndarray):
        """Gets the expected probabilities of each
        category of y based on the data"""
        if len(y) == 0:
            return {name: 0.0 for name in self.class_names}

        unique_vals, counts = np.unique(y, return_counts=True)
        prob_dict = {
            str(val): count / len(y) for val, count in zip(unique_vals, counts)
        }

        # Fill in missing classes with 0 probability
        for class_name in self.class_names:
            if class_name not in prob_dict:
                prob_dict[class_name] = 0.0

        return prob_dict

    def predict(self, X: pd.DataFrame):
        """Predict the category of each sample in X"""
        assert (
            list(X.columns) == self.feature_names
        ), "Inference data does not have the same columns as train data"

        X_np = X.values
        n_samples = len(X)
        n_classes = len(self.class_names)

        # Pre-allocate result array
        predictions = np.zeros((n_samples, n_classes))
        sample_indices = np.arange(n_samples)

        nodes_to_visit = [(self.root, sample_indices)]

        while nodes_to_visit:
            node, indices = nodes_to_visit.pop()

            if node.split_column is None:
                # Leaf node - assign probabilities
                for i, class_name in enumerate(self.class_names):
                    predictions[indices, i] = node.probabilities.get(class_name, 0)
            else:
                # Split node
                col_idx = self.feature_names.index(node.split_column)
                left_mask, right_mask = node.split(X_np[indices], col_idx)

                left_indices = indices[left_mask]
                right_indices = indices[right_mask]

                if len(left_indices) > 0:
                    nodes_to_visit.append((node.left, left_indices))
                if len(right_indices) > 0:
                    nodes_to_visit.append((node.right, right_indices))

        return pd.DataFrame(predictions, columns=self.class_names, index=X.index)

    def fit(self, pbar=None):
        """Expand the tree fully using the data from
        X_train and y_train"""
        if self.use_progress_bar and pbar is None:
            pbar = tqdm(total=1.0, desc="Training RandCart")

        def should_stop(y, best_gain, depth):
            """Returns a boolean with the condition to stop expanding .
            Stops if all labels are the same , gain is too low ,
            or max depth has been reached ."""
            return (
                len(y) < self.min_samples_split
                or gini_impurity(y) == 0
                or best_gain <= 0
                or depth >= self.max_depth
            )

        def expand_tree(node: CartNode, indices: np.ndarray, depth: int):
            """Expand the node with the data described by indices ,
            choose the best gini split using the randomized split search.
            Don ’t expand if the conditions are met"""
            X_temp = self.X_train_np[indices]
            y_temp = self.y_train_np[indices]

            # Randomly select features to explore
            n_features_to_try = min(
                self.max_features_to_explore, len(self.feature_names)
            )
            random_feature_indices = np.random.choice(
                len(self.feature_names), n_features_to_try, replace=False
            )
            random_features = [self.feature_names[i] for i in random_feature_indices]

            best_splits = gini_best_splits(
                X_temp, y_temp, self.dtypes, self.feature_names, random_features
            )

            if should_stop(y_temp, best_splits[0][2], depth):
                if pbar is not None:
                    pbar.update(
                        (2 ** (self.max_depth + 1 - depth) - 1)
                        / (2 ** (self.max_depth + 1) - 1)
                    )
                return

            chosen_split = random.choice(best_splits)
            if pbar is not None:
                pbar.update(1 / (2 ** (self.max_depth + 1) - 1))

            node.split_column = chosen_split[0]
            node.condition = chosen_split[1]

            col_idx = self.feature_names.index(node.split_column)
            left_mask, right_mask = node.split(X_temp, col_idx)

            left_indices = indices[left_mask]
            left_y = y_temp[left_mask]
            node.left = CartNode(None, None, None, None, self.get_probs_as_dict(left_y))
            expand_tree(node.left, left_indices, depth + 1)

            right_indices = indices[right_mask]
            right_y = y_temp[right_mask]
            node.right = CartNode(
                None, None, None, None, self.get_probs_as_dict(right_y)
            )
            expand_tree(node.right, right_indices, depth + 1)

        initial_indices = np.arange(len(self.y_train_np))
        expand_tree(self.root, initial_indices, 0)

        if self.use_progress_bar:
            pbar.close()

    def print_tree(self):
        def print_node(node: CartNode, prefix: str = "", is_left: bool = True):
            connector = "├── " if is_left else "└── "
            if node.split_column is None:
                probs = ", ".join(
                    f"{k}: {v:.2f}" for k, v in node.probabilities.items()
                )
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
