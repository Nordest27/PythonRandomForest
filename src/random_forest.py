import numpy as np

from class_reg_decision_tree import RandCart


def predict_rf(tree_ls, X_test):
    # One row per each entry in the X_test
    predictions = [[] for _ in range(len(X_test))]
    for tree in tree_ls:
        tree_pred = tree.predict(X_test)

        for entry in range(len(tree_pred)):
            predictions[entry].append(tree_pred.iloc[entry].idxmax())

    # Get the max prediction of each tree for each element in X_test
    preds = []
    for row_pred in predictions:
        counter = [[x, row_pred.count(x)] for x in set(row_pred)]
        counter_sorted = sorted(counter, key=lambda item: -item[1])
        preds.append(counter_sorted[0][0])

    return preds


def draw_bootstrap(X_train, y_train, bootstrap_size):
    bootstrap_indices = np.random.randint(len(X_train), size=bootstrap_size)
    oob_indices = np.array(
        [i for i in range(len(X_train)) if i not in bootstrap_indices]
    )

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


def oob_score(tree, X_test, y_test):
    mis_label = 0
    pred = tree.predict(X_test)
    for i in range(len(X_test)):
        if pred.iloc[i].idxmax() != y_test.iloc[i]:
            mis_label += 1
    return mis_label / len(X_test)


def random_forest(
    X_train,
    y_train,
    n_estimators,
    max_features,
    max_depth,
    bootstrap_size,
    min_samples_split,
):
    tree_ls = list()
    oob_ls = list()
    for i in range(n_estimators):
        print(f"Creating tree #{i}")
        X_bootstrap, y_bootstrap, X_oob, y_oob = draw_bootstrap(
            X_train, y_train, bootstrap_size
        )
        tree = RandCart(
            X_bootstrap, y_bootstrap, max_features, max_depth, min_samples_split, True
        )
        tree.fit()
        tree_ls.append(tree)
        oob_error = oob_score(tree, X_oob, y_oob)
        oob_ls.append(oob_error)
    print("Random forest OOB estimate: {:.2f}".format(np.mean(oob_ls)))
    return tree_ls
