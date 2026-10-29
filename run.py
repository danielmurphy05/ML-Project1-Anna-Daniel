"""run.py: reproduces the submission file with regularized logistic regression.

Pipeline:
    1. Load data with the provided helper (load_csv_data).
    2. Apply fixed cleaning rules to the whole dataset (no learning involved).
    3. Stratified 60/20/20 split into train / validation / internal test.
    4. Fit the preprocessing on the train set only, then transform val / test.
    5. Tune lambda on validation, then pick the decision threshold on validation.
    6. Evaluate once on the internal test set.
    7. Predict x_test.csv and write the sample-submission.csv with create_csv_submission.
"""

import numpy as np

from helpers import create_csv_submission, load_csv_data
from implementations import (
    compute_logistic_loss,
    reg_logistic_regression,
    sigmoid,
)
from preprocessing import clean_fixed, fit_preprocessing, transform_preprocessing

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_PATH = "dataset/"
SUBMISSION_PATH = "sample-submission.csv"

SEED = 42
LAMBDAS = [0, 1e-5, 1e-4, 1e-3]          # regularization grid
GAMMAS = [0.1, 0.3, 0.8]                 # step sizes to try
CHECKPOINTS = [500, 1000, 2000]          # max_iters values to compare
THRESHOLDS = np.linspace(0.02, 0.9, 177) # candidate decision thresholds


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def add_bias(X):
    """Prepend a column of ones (the bias term) to X."""
    return np.c_[np.ones(len(X)), X]


def metrics(y_true, y_pred):
    """Return accuracy, precision, recall, F1 and (tp, fp, fn, tn)."""
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    acc = (tp + tn) / len(y_true)
    prec = tp / (tp + fp) if tp + fp > 0 else 0
    rec = tp / (tp + fn) if tp + fn > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0
    return acc, prec, rec, f1, (tp, fp, fn, tn)


def stratified_cut(idx, a, b):
    """Split a shuffled index array into [0, a), [a, b) and [b, 1) fractions."""
    n = len(idx)
    return idx[: int(a * n)], idx[int(a * n): int(b * n)], idx[int(b * n):]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    # ------------------------------------------------------------------
    # 1. Load data with the provided helper.
    #    load_csv_data returns the features WITHOUT the Id column, the labels
    #    in {-1, 1}, and the ids of the train and test sets.
    # ------------------------------------------------------------------
    x_train, x_test, y_train_raw, train_ids, test_ids = load_csv_data(DATA_PATH)

    # Column names (the helper drops the Id column, so we drop it from the header too)
    header = np.genfromtxt(DATA_PATH + "x_train.csv", delimiter=",", max_rows=1, dtype=str)
    assert header[0] == "Id", "First column of x_train.csv should be Id"
    columns = header[1:]
    assert x_train.shape[1] == len(columns), "Header and data do not match"

    # Convert labels from {-1, 1} to {0, 1} (required by the logistic loss)
    y = (np.asarray(y_train_raw) == 1).astype(float)
    assert len(x_train) == len(y), "X and y must have the same number of rows"
    print("Classes:", np.unique(y, return_counts=True))

    # ------------------------------------------------------------------
    # 2. Fixed cleaning rules (drop columns by name, special codes -> NaN/0)
    # ------------------------------------------------------------------
    data_clean, cols = clean_fixed(x_train, columns)

    # ------------------------------------------------------------------
    # 3. Stratified 60 / 20 / 20 split (same positive rate in every subset)
    # ------------------------------------------------------------------
    rng = np.random.default_rng(SEED)
    idx_pos = rng.permutation(np.where(y == 1)[0])
    idx_neg = rng.permutation(np.where(y == 0)[0])

    pos_tr, pos_va, pos_te = stratified_cut(idx_pos, 0.6, 0.8)
    neg_tr, neg_va, neg_te = stratified_cut(idx_neg, 0.6, 0.8)

    train_idx = rng.permutation(np.concatenate([pos_tr, neg_tr]))
    val_idx = rng.permutation(np.concatenate([pos_va, neg_va]))
    test_idx = rng.permutation(np.concatenate([pos_te, neg_te]))

    y_tr, y_val, y_te = y[train_idx], y[val_idx], y[test_idx]
    print("Positive rate (train/val/test):", y_tr.mean().round(3), y_val.mean().round(3), y_te.mean().round(3))

    # ------------------------------------------------------------------
    # 4. Preprocessing: fit on train only, transform val and test
    # ------------------------------------------------------------------
    X_tr, cols_final, params = fit_preprocessing(data_clean[train_idx], cols)
    X_val = transform_preprocessing(data_clean[val_idx], params)
    X_te = transform_preprocessing(data_clean[test_idx], params)
    print("Final number of features:", X_tr.shape[1])

    # Add the bias column AFTER preprocessing (otherwise it would be dropped
    # as a constant column)
    tx_tr = add_bias(X_tr)
    tx_val = add_bias(X_val)
    tx_te = add_bias(X_te)

    # ------------------------------------------------------------------
    # 5a. Tune lambda, gamma and max_iters on the validation set.
    #     Only the best result is printed (not every combination).
    # ------------------------------------------------------------------
    initial_w = np.zeros(tx_tr.shape[1])

    best = {"loss": np.inf}
    for lambda_ in LAMBDAS:
        for gamma in GAMMAS:
            w_cur = initial_w
            done = 0
            for ckpt in CHECKPOINTS:
                w_cur, loss_tr = reg_logistic_regression(
                    y_tr, tx_tr, lambda_, w_cur, ckpt - done, gamma
                )
                done = ckpt
                loss_va = compute_logistic_loss(y_val, tx_val, w_cur)

                if not np.isfinite(loss_va):      # diverged: stop this run
                    break
                if loss_va < best["loss"]:
                    best = {"loss": loss_va, "lambda": lambda_, "gamma": gamma,
                            "iters": ckpt, "w": w_cur.copy(), "loss_tr": loss_tr}

    print(f"Best: lambda={best['lambda']} gamma={best['gamma']} "
          f"iters={best['iters']} | loss train {best['loss_tr']:.5f} "
          f"| loss val {best['loss']:.5f}")

    # Best weights found (last w of that run, loss without the penalty term)
    w = best["w"]

    # Best weights found (last w of that run, loss without the penalty term)
    w = best["w"]
    print("Final train loss:", round(best["loss_tr"], 4))

    # ------------------------------------------------------------------
    # 5b. Choose the decision threshold on the validation set (max F1)
    # ------------------------------------------------------------------
    p_val = sigmoid(tx_val @ w)
    f1s = [metrics(y_val, (p_val >= t).astype(int))[3] for t in THRESHOLDS]
    best_t = THRESHOLDS[int(np.argmax(f1s))]

    acc, prec, rec, f1, _ = metrics(y_val, (p_val >= best_t).astype(int))
    print(f"VAL  -> threshold {best_t:.3f} | Acc {acc:.3f} | " f"Prec {prec:.3f} | Rec {rec:.3f} | F1 {f1:.3f}")

    # ------------------------------------------------------------------
    # 6. Internal test: evaluate once
    # ------------------------------------------------------------------
    p_te = sigmoid(tx_te @ w)
    acc, prec, rec, f1, conf = metrics(y_te, (p_te >= best_t).astype(int))
    print(f"TEST -> Acc {acc:.3f} | Prec {prec:.3f} | Rec {rec:.3f} | F1 {f1:.3f}")
    print("TP, FP, FN, TN:", conf)

    # ------------------------------------------------------------------
    # 7. Predict x_test.csv and write the submission file.
    #    test_ids comes from load_csv_data, so the indexes match x_test.csv.
    # ------------------------------------------------------------------
    x_test_clean, _ = clean_fixed(x_test, columns)
    X_new = transform_preprocessing(x_test_clean, params)

    p_new = sigmoid(add_bias(X_new) @ w)
    y_new = (p_new >= best_t).astype(int)         # predictions in {0, 1}
    y_pred = np.where(y_new == 1, 1, -1)          # submission format: {-1, 1}

    assert len(test_ids) == len(y_pred), "Ids and predictions are not aligned"
    create_csv_submission(test_ids, y_pred, SUBMISSION_PATH)

    print(f"Submission written to {SUBMISSION_PATH}")
    print("Predicted values:", np.unique(y_pred, return_counts=True))
    print("Predicted positive rate:", (y_new == 1).mean())


if __name__ == "__main__":
    main()