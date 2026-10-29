
# ML Project 1 – Predicting coronary heart disease (MICHD)

EPFL CS-433 Machine Learning, Project 1. Team: Anna Vandermersch, Daniel Murphy.

We predict whether a person has coronary heart disease or has had a heart attack
(MICHD) from their answers to the BRFSS health survey. Everything is implemented
with NumPy only (no pandas, scikit-learn, or other ML libraries).

## Repository structure

| File | Content |
|------|---------|
| `implementations.py` | The six required methods: `mean_squared_error_gd`, `mean_squared_error_sgd`, `least_squares`, `ridge_regression`, `logistic_regression`, `reg_logistic_regression`, plus their loss/gradient helpers. |
| `preprocessing.py` | Data cleaning and preprocessing (see below). |
| `run.py` | Reproduces our final submission: trains regularized logistic regression and writes the predictions. |
| `compare.py` | Baselines and ablation study used in the report (method comparison and pipeline ablation). |
| `helpers.py` | Course-provided helpers to load the CSV data and write the submission file. |
| `sample-submission.csv` | Our final predictions, as produced by `run.py`. |

## Setup

1. Python 3.9+ with NumPy installed (`pip install numpy`). No other library is needed.
2. Download the data from the AIcrowd challenge (or `projects/project1/data/dataset_to_release.zip`
   in the course repository) and unzip it so that the three CSV files are in a `dataset/`
   folder at the root of this repository:
   ```
   dataset/x_train.csv
   dataset/y_train.csv
   dataset/x_test.csv
   ```
   Do not rename the files.

## Reproducing our results

```bash
python run.py       # writes sample-submission.csv (our final AIcrowd submission)
python compare.py   # prints the method comparison and ablation tables, saves comparison_results.csv
```

`run.py` uses a fixed random seed (42), so it produces exactly the same predictions every time.
It takes about XX minutes on a laptop, most of it for the hyperparameter search.

## Pipeline

1. **Fixed cleaning** (`clean_fixed`): drop administrative and survey-logistics columns
   (dates, identifiers, phone checks), then decode survey special codes. Values starting
   with 7 or 9 ("don't know" / "refused") become missing, and values starting with 8 or 5
   ("none") become 0, but only when the value is isolated in its column, so real values
   such as 7 days or an age of 77 are kept. Codes are detected on the training data and
   applied identically to the test data.
2. **Split**: stratified 60/20/20 train / validation / internal test split (same positive
   rate, about 8.8%, in each part).
3. **Learned preprocessing** (`fit_preprocessing`, fitted on the training split only, then
   applied with `transform_preprocessing`): recode binary answers {1, 2} to {1, 0}, drop
   features with more than 50% missing values, impute medians, remove constant and highly
   correlated (|r| > 0.99) features, standardize non-binary features and clip them to
   [-5, 5]. A bias column is added afterwards.
4. **Model**: regularized logistic regression. λ, the step size γ and the number of
   iterations are chosen on the validation set (lowest validation loss).
5. **Decision threshold**: chosen on the validation set to maximize F1, since the classes
   are imbalanced and the competition metric is F1.
6. **Evaluation**: the internal test split is used once, for the final reported scores.
   Predictions on `x_test.csv` are then written with `create_csv_submission`.