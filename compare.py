"""compare.py: baselines and ablation study for the report.

Uses exactly the same cleaning, stratified 60/20/20 split and preprocessing as
run.py, then
  1. compares the methods from implementations.py (plus a trivial baseline),
  2. runs an ablation study on our final model (regularized logistic regression),
     removing one ingredient of the pipeline at a time.

For every model the decision threshold is chosen on the validation set (max F1),
and the scores are reported on the internal test set, which is used only once.

Usage:  python compare.py      (expects the data in dataset/, like run.py)
"""

import time

import numpy as np

import preprocessing as prep
from helpers import load_csv_data
from implementations import (
    least_squares,
    logistic_regression,
    mean_squared_error_gd,
    mean_squared_error_sgd,
    reg_logistic_regression,
    ridge_regression,
)
from preprocessing import (
    clean_fixed,
    find_special_codes,
    fit_preprocessing,
    transform_preprocessing,
)
from run import DATA_PATH, SEED, add_bias, metrics, stratified_cut

# Hyperparameters of the final model. Set them to the values printed by run.py
# ("Best: lambda=... gamma=... iters=...").
# run.py selects lambda = 0, i.e. plain logistic regression.
BEST_LAMBDA = 0.0
BEST_GAMMA = 0.8
BEST_ITERS = 2000

# Settings for the other methods
GD_GAMMA, GD_ITERS = 0.1, 1000  # linear regression, full gradient descent
SGD_GAMMA, SGD_ITERS = 1e-3, 200000  # linear regression, SGD (1 sample per step)
RIDGE_LAMBDAS = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]  # ridge: best lambda chosen on val
REG_LAMBDA = 1e-4  # regularized logistic row, to show the effect of the penalty


# ---------------------------------------------------------------------------
# Data preparation (same steps as run.py)
# ---------------------------------------------------------------------------
def load():
    """Load the training data, its column names and the labels in {0, 1}."""
    x_train, _, y_raw, _, _ = load_csv_data(DATA_PATH)
    header = np.genfromtxt(
        DATA_PATH + "x_train.csv", delimiter=",", max_rows=1, dtype=str
    )
    y = (np.asarray(y_raw) == 1).astype(float)
    return x_train, header[1:], y


def split(y):
    """Stratified 60/20/20 split, identical to the one in run.py."""
    rng = np.random.default_rng(SEED)
    idx_pos = rng.permutation(np.where(y == 1)[0])
    idx_neg = rng.permutation(np.where(y == 0)[0])
    pos = stratified_cut(idx_pos, 0.6, 0.8)
    neg = stratified_cut(idx_neg, 0.6, 0.8)
    return [rng.permutation(np.concatenate([p, n])) for p, n in zip(pos, neg)]


def prepare(x, columns, y, drop_list=prep.DROP_LIST, use_codes=True, bias=True):
    """Clean, split and preprocess. Returns (tx_tr, y_tr, tx_val, y_val, tx_te, y_te).

    use_codes=False skips the special-code replacement (7/77/9/99 -> NaN, 8/88 -> 0).
    bias=False leaves out the column of ones."""
    codes = None if use_codes else {}
    data, cols = clean_fixed(x, columns, drop_list=drop_list, codes=codes)
    tr, va, te = split(y)
    X_tr, _, params = fit_preprocessing(data[tr], cols)
    X_va = transform_preprocessing(data[va], params)
    X_te = transform_preprocessing(data[te], params)
    f = add_bias if bias else (lambda X: X)
    return f(X_tr), y[tr], f(X_va), y[va], f(X_te), y[te]


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def best_threshold(s_val, y_val):
    """Threshold on the scores that maximizes F1 on the validation set.
    Candidates are quantiles of the validation scores, so this works for any
    score scale (probabilities or raw linear-regression outputs)."""
    cands = np.quantile(s_val, np.linspace(0.5, 0.995, 200))
    f1s = [metrics(y_val, (s_val >= t).astype(int))[3] for t in cands]
    return cands[int(np.argmax(f1s))]


def evaluate(name, w, d, threshold=None):
    """Score a weight vector on the internal test set and return a result row.
    threshold=None tunes the threshold on validation; otherwise it is fixed."""
    tx_tr, y_tr, tx_va, y_va, tx_te, y_te = d
    s_va, s_te = tx_va @ w, tx_te @ w
    t = best_threshold(s_va, y_va) if threshold is None else threshold
    acc, prec, rec, f1, _ = metrics(y_te, (s_te >= t).astype(int))
    print(f"  {name:<40s} F1 {f1:.3f}  Acc {acc:.3f}  Prec {prec:.3f}  Rec {rec:.3f}")
    return name, f1, acc, prec, rec


def fit_final(d, lambda_=BEST_LAMBDA):
    """Train the final model (regularized logistic regression)."""
    tx_tr, y_tr = d[0], d[1]
    w, _ = reg_logistic_regression(
        y_tr, tx_tr, lambda_, np.zeros(tx_tr.shape[1]), BEST_ITERS, BEST_GAMMA
    )
    return w


# ---------------------------------------------------------------------------
# 1. Method comparison
# ---------------------------------------------------------------------------
def compare_methods(d):
    tx_tr, y_tr, tx_va, y_va = d[:4]
    y_pm = 2 * y_tr - 1  # linear regression is trained on labels in {-1, 1}
    w0 = np.zeros(tx_tr.shape[1])
    rows = []

    # Trivial baseline: always predict "no disease"
    acc = 1 - d[5].mean()
    print(f"  {'Always predict -1':<40s} F1 0.000  Acc {acc:.3f}")
    rows.append(("Always predict -1", 0.0, acc, 0.0, 0.0))

    w, _ = mean_squared_error_gd(y_pm, tx_tr, w0, GD_ITERS, GD_GAMMA)
    rows.append(evaluate("Linear regression (GD)", w, d))

    np.random.seed(SEED)  # SGD samples one random point per step
    w, _ = mean_squared_error_sgd(y_pm, tx_tr, w0, SGD_ITERS, SGD_GAMMA)
    rows.append(evaluate("Linear regression (SGD)", w, d))

    w, _ = least_squares(y_pm, tx_tr)
    rows.append(evaluate("Least squares", w, d))

    # Ridge: choose lambda by validation F1
    best = None
    for lam in RIDGE_LAMBDAS:
        w, _ = ridge_regression(y_pm, tx_tr, lam)
        t = best_threshold(tx_va @ w, y_va)
        f1 = metrics(y_va, (tx_va @ w >= t).astype(int))[3]
        if best is None or f1 > best[0]:
            best = (f1, lam, w)
    rows.append(evaluate(f"Ridge regression (lambda={best[1]:g})", best[2], d))

    w_log, _ = logistic_regression(y_tr, tx_tr, w0, BEST_ITERS, BEST_GAMMA)
    rows.append(evaluate("Logistic regression (final model)", w_log, d))

    w = fit_final(d, lambda_=REG_LAMBDA)
    rows.append(evaluate(f"Reg. logistic regression (lambda={REG_LAMBDA:g})", w, d))
    return rows, w_log


# ---------------------------------------------------------------------------
# 2. Ablation study on the final model
# ---------------------------------------------------------------------------
def ablation(x, columns, y, d_full, w_full):
    rows = [evaluate("Full pipeline", w_full, d_full)]

    # Default threshold 0.5 on the probability = 0 on the linear score
    rows.append(evaluate("No threshold tuning (p >= 0.5)", w_full, d_full, 0.0))

    d = prepare(x, columns, y, bias=False)
    rows.append(evaluate("No bias term", fit_final(d), d))

    d = prepare(x, columns, y, use_codes=False)
    rows.append(evaluate("No special-code cleaning", fit_final(d), d))

    old_clip = prep.CLIP
    prep.CLIP = np.inf  # transform/fit read prep.CLIP when they run
    d = prepare(x, columns, y)
    prep.CLIP = old_clip
    rows.append(evaluate("No clipping of outliers", fit_final(d), d))

    
    # Reinsertion of columns one by one 
    for col in prep.DROP_LIST:
        drop_list = [c for c in prep.DROP_LIST if c != col]

        d = prepare(x, columns, y, drop_list=drop_list)

        rows.append(
            evaluate(f"Keep {col}", fit_final(d), d)
        )
    return rows


def main():
    t0 = time.time()
    x, columns, y = load()
    print("Preparing data...")
    d = prepare(x, columns, y)
    print("Features (with bias):", d[0].shape[1])

    print("\n=== Method comparison (internal test set) ===")
    rows_m, w_full = compare_methods(d)

    print("\n=== Ablation of the final model (internal test set) ===")
    rows_a = ablation(x, columns, y, d, w_full)

    # Save both tables for the report
    with open("comparison_results.csv", "w") as f:
        f.write("table,setting,f1,accuracy,precision,recall\n")
        for tab, rows in (("methods", rows_m), ("ablation", rows_a)):
            for r in rows:
                f.write(f"{tab},{r[0]},{r[1]:.4f},{r[2]:.4f},{r[3]:.4f},{r[4]:.4f}\n")
    print(f"\nSaved comparison_results.csv ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
