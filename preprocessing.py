import numpy as np

DROP_LIST = [
    'Id', '_PSU', 'DISPCODE', 'SEQNO', 'IYEAR', 'IMONTH',
    'IDAY', 'IDATE', 'FMONTH', 'CELLFON3', 'LANDLINE',
    'SEX', 'PVTRESD2', 'CSTATE', 'CTELENUM1', 'CTELENUM',
    'CTELNUM1', 'CELLFON2', 'PVTRESD1', '_VEGETEX',
    '_VEGRESP', '_FRTRESP', '_FRUITEX', 'CVDINFR4', 'CVDCRHD4'
]  # columns related to phone-survey checks, and columns too informative about the target (CVDINFR4, CVDCRHD4: data leakage)

# Special codes are detected by their leading digits, whatever their length:
# 7, 77, 777, ... and 9, 99, 999, ... ("don't know" / "refused") -> NaN
# 8, 88, 888, ... and 5, 55, 555, ...  ("none")                   -> 0
# (a prefix matches any number starting with it: prefix 7 matches 7, 70-79, 700-799, ...)
PREFIXES_TO_NAN = [7, 9]
PREFIXES_TO_ZERO = [8, 5]
MAX_DIGITS = 10               # longest number checked for a given prefix

# A value matching a prefix is only treated as a code if it is ISOLATED in its
# column: no other value of the column lies within max(1, REL_GAP * value) of it.
# Examples: 77 in PHYSHLTH (values 0-30, 88, 99) is a code; 7 or 8 days in the same
# column are real (6 and 9 exist next to them); 77 in an age column or 5512 in a
# BMI column are real values (their neighbours exist).
REL_GAP = 0.1

CLIP = 5                        # clipping bound after standardization (in std units)


def prefix_hit(values, prefixes):
    """For an array of DISTINCT values, return a boolean array telling which ones
    are positive integers whose leading digits equal one of the prefixes
    (prefix 77 matches 77, 770, 7712, ...). NaN and non-integers never match."""
    values = np.asarray(values, dtype=float)
    ok = np.isfinite(values) & (values > 0) & (values == np.floor(values))
    hit = np.zeros(len(values), dtype=bool)
    for p in prefixes:
        n_p = len(str(p))
        for n_digits in range(n_p, MAX_DIGITS + 1):
            low = p * 10 ** (n_digits - n_p)
            high = (p + 1) * 10 ** (n_digits - n_p)
            hit |= ok & (values >= low) & (values < high)
    return hit


def find_special_codes(data, columns, isolated_only=True, rel_gap=REL_GAP):
    """Find, column by column, which values are special codes.

    Returns a dict {column name: {"nan": [values -> NaN], "zero": [values -> 0]}}.
    With isolated_only=True a value matching a prefix is kept as a code only if it
    is far from every other value of its column; with False, every value that
    matches a prefix is a code (the plain prefix rule).

    Call it ONCE on the training features, then pass the result to clean_fixed for
    the train AND the test data, so both are cleaned with the same decisions."""
    data = np.asarray(data, dtype=float)
    codes, rejected = {}, {}

    for j, name in enumerate(columns):
        col = data[:, j]
        vals = np.unique(col[np.isfinite(col)])
        hit_nan = prefix_hit(vals, PREFIXES_TO_NAN)
        hit_zero = prefix_hit(vals, PREFIXES_TO_ZERO)
        cand = hit_nan | hit_zero
        if not cand.any():
            continue

        is_code = cand.copy()
        if isolated_only:
            # Distance of each candidate to its nearest OTHER distinct value
            # (vals is sorted, so the neighbours are the previous and next entries)
            cidx = np.where(cand)[0]
            cv = vals[cidx]
            left = np.where(cidx > 0, vals[np.maximum(cidx - 1, 0)], -np.inf)
            right = np.where(cidx < len(vals) - 1,
                             vals[np.minimum(cidx + 1, len(vals) - 1)], np.inf)
            dist = np.minimum(cv - left, right - cv)
            iso = dist > np.maximum(1.0, rel_gap * cv)
            is_code = np.zeros(len(vals), dtype=bool)
            is_code[cidx[iso]] = True
            if (~iso).any():
                rejected[str(name)] = cv[~iso].astype(int).tolist()

        nan_vals = vals[is_code & hit_nan]
        zero_vals = vals[is_code & hit_zero]
        if len(nan_vals) or len(zero_vals):
            codes[str(name)] = {"nan": nan_vals.tolist(), "zero": zero_vals.tolist()}

    print("Columns with special codes:", len(codes))
    for name, r in codes.items():
        print(f"  {name}: NaN {r['nan'][:6]} | zero {r['zero'][:6]}")
    if rejected:
        print("Kept as real values (not isolated):")
        for name, v in rejected.items():
            print(f"  {name}: {v[:8]}")
    return codes


# ======================================================
# PART 1: fixed rules -> applied to the WHOLE dataset
# (and to the new test data, in exactly the same way)
# ======================================================
def clean_fixed(data, columns, drop_list=DROP_LIST, codes=None):
    """Drop useless columns and replace the special codes.
    codes: result of find_special_codes on the training features. If None, the
    codes are found on the data passed in (fine for the training data only)."""
    data = np.asarray(data, dtype=float).copy()
    columns = np.asarray(columns)

    # 1. Remove useless columns (by name)
    keep = np.array([c not in drop_list for c in columns])
    print("Removed columns:", columns[~keep].tolist())
    data, columns = data[:, keep], columns[keep]

    # 2. Special codes
    if codes is None:
        codes = find_special_codes(data, columns)
    for j, name in enumerate(columns):
        rule = codes.get(str(name))
        if rule is None:
            continue
        col = data[:, j]
        mask_nan = np.isin(col, rule["nan"])
        mask_zero = np.isin(col, rule["zero"])
        col[mask_nan] = np.nan
        col[mask_zero] = 0

    return data, columns


# ======================================================
# PART 2: everything that learns -> fit on the TRAIN set only
# ======================================================
def fit_preprocessing(X_train, columns, miss_thr=0.5, corr_thr=0.99):
    X = X_train.copy()
    columns = np.asarray(columns)
    idx = np.arange(X.shape[1])          # positions of the original columns

    # 1. Binary columns {1, 2} -> {1, 0}  (before imputation)
    bin_mask = np.array([
        set(np.unique(X[~np.isnan(X[:, j]), j])) == {1.0, 2.0}
        for j in range(X.shape[1])
    ])
    X[:, bin_mask] = np.where(X[:, bin_mask] == 2, 0, X[:, bin_mask])

    # 2. Remove columns with too many NaN
    keep = np.isnan(X).mean(axis=0) <= miss_thr
    idx, X, bin_mask = idx[keep], X[:, keep], bin_mask[keep]

    # 3. Median imputation (medians computed on the train set)
    medians = np.nanmedian(X, axis=0)
    X = np.where(np.isnan(X), medians, X)

    # 4. Constant columns
    keep = X.std(axis=0) > 0
    idx, X, bin_mask, medians = idx[keep], X[:, keep], bin_mask[keep], medians[keep]
    print("Constant columns removed:", (~keep).sum())

    # 5. Highly correlated columns
    if X.shape[1] > 1:
        upper = np.triu(np.abs(np.corrcoef(X, rowvar=False)), k=1)
        keep = ~np.any(upper > corr_thr, axis=0)
        print("Columns removed for correlation:", columns[idx][~keep].tolist())
        idx, X, bin_mask, medians = idx[keep], X[:, keep], bin_mask[keep], medians[keep]

    # 6. Standardization (non-binary columns only)
    cont = ~bin_mask
    mean = X[:, cont].mean(axis=0)
    std = X[:, cont].std(axis=0)
    std[std == 0] = 1.0
    X[:, cont] = (X[:, cont] - mean) / std

    # 7. Clip standardized values to [-CLIP, CLIP] to limit the effect of outliers
    X[:, cont] = np.clip(X[:, cont], -CLIP, CLIP)

    params = {"idx": idx, "bin_mask": bin_mask, "medians": medians,
              "mean": mean, "std": std}
    return X, columns[idx], params


def transform_preprocessing(X_new, params):
    """Apply the parameters learned on the train set to the val / test data."""
    X = X_new[:, params["idx"]].copy()

    b = params["bin_mask"]
    X[:, b] = np.where(X[:, b] == 2, 0, X[:, b])
    X = np.where(np.isnan(X), params["medians"], X)

    cont = ~b
    X[:, cont] = (X[:, cont] - params["mean"]) / params["std"]

    # Same clipping as in fit_preprocessing
    X[:, cont] = np.clip(X[:, cont], -CLIP, CLIP)
    return X