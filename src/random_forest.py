import numpy as np
from tqdm import tqdm
from class_reg_decision_tree import RandCart


class RandomForest:

    @staticmethod
    def draw_bootstrap(X_train, y_train, bootstrap_size, show=False):
        bootstrap_indices = np.random.randint(len(X_train), size=bootstrap_size)
        oob_indices = np.array(
            [i for i in range(len(X_train)) if i not in bootstrap_indices]
        )
        if show:
            print(f"Bootstrap size: {len(bootstrap_indices)}")
            print(f"OOB size: {len(oob_indices)}")
            print(
                f"Duplicated elements: {(len(oob_indices)+len(bootstrap_indices)-len(X_train))}"
            )

        X_bootstrap = X_train.iloc[bootstrap_indices]
        y_bootstrap = y_train.iloc[bootstrap_indices]
        X_oob = X_train.iloc[oob_indices]
        y_oob = y_train.iloc[oob_indices]

        return X_bootstrap, y_bootstrap, X_oob, y_oob

    def __init__(
        self,
        n_estimators: int,
        max_features: int,
        max_depth: int,
        bootstrap_size: int,
        min_samples_split: int,
    ):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.max_depth = max_depth
        self.bootstrap_size = bootstrap_size
        self.min_samples_split = min_samples_split
        self.tree_ls = list()

    def oob_score(self, i, X_test, y_test):
        mis_label = 0
        pred = self.tree_ls[i].predict(X_test)
        for i in range(len(X_test)):
            if pred.iloc[i].idxmax() != y_test.iloc[i]:
                mis_label += 1
        return mis_label / len(X_test)

    def predict(self, X_test):
        # One row per each entry in the X_test
        predictions = [[] for _ in range(len(X_test))]
        for tree in self.tree_ls:
            tree_pred = tree.predict(X_test)

            for entry in range(len(tree_pred)):
                predictions[entry].append(tree_pred.iloc[entry].idxmax())

        # Get the max prediction of each tree for each element in X_test
        preds = []
        for row_pred in predictions:
            counter = [[x, row_pred.count(x)] for x in set(row_pred)]
            counter_sorted = sorted(counter, key=lambda item: -item[1])
            preds.append(counter_sorted[0][0])

        return np.array(preds)

    def fit(self, X_train: np.array, y_train: np.array, show: str | None = None):
        if show not in ["trees", "progress", None]:
            raise ValueError("Parameter 'show' can be 'trees', 'progress' or None")
        pbar = None
        if show == "progress":
            pbar = tqdm(total=self.n_estimators, desc="Training Random Forest")

        self.tree_ls = list()
        oob_ls = list()
        for i in range(self.n_estimators):
            if show == "trees":
                print(f"Creating tree #{i}")
            X_bootstrap, y_bootstrap, X_oob, y_oob = self.draw_bootstrap(
                X_train, y_train, self.bootstrap_size, show == "trees"
            )
            tree = RandCart(
                X_bootstrap,
                y_bootstrap,
                self.max_features,
                self.max_depth,
                self.min_samples_split,
                show == "trees",
            )
            tree.fit(pbar)
            self.tree_ls.append(tree)
            oob_error = self.oob_score(-1, X_oob, y_oob)
            oob_ls.append(oob_error)
        if show == "progress":
            pbar.close()
        print("Random forest OOB estimate: {:.2f}".format(np.mean(oob_ls)))
