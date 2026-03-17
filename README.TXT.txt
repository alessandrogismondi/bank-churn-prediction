# Bank Customer Churn Prediction

This project focuses on predicting whether a bank customer is likely to leave the bank (churn) based on demographic and account-related information. The goal is to build a simple, clear baseline model using scikit-learn and to document the full workflow from raw data to evaluation.

## Dataset

The dataset contains 10,000 bank customers with 18 original features, including:

- Customer demographics (age, gender, geography)
- Account information (balance, tenure, number of products, credit card ownership, activity status)
- Additional attributes (card type, satisfaction score, complaints, points earned)
- Target variable: `Exited` (0 = customer stayed, 1 = customer churned)

Missing values are not present in this dataset, and several identifier columns are removed before modeling.

## Project structure

- `notebooks/`
  - `bank_churn_ml.ipynb` – main notebook with the full analysis and modeling
- `data/` (optional, not pushed if the dataset is private)
- `README.md` – project description and usage instructions

## Methodology

The modeling pipeline follows these main steps:

1. **Exploratory data inspection** using `pandas` (`info`, `describe`, and basic checks for missing values).
2. **Feature preparation**: removal of identifier fields (`RowNumber`, `CustomerId`, `Surname`), definition of features `X` and target `y` (`Exited`).
3. **Train–test split** with an 80/20 ratio and stratification on the target to preserve the churn rate in both sets.
4. **Preprocessing and modeling** using a scikit-learn `Pipeline`:
   - `ColumnTransformer` with `OneHotEncoder` for categorical variables (`Geography`, `Gender`, `Card Type`) and passthrough for numerical features.
   - `LogisticRegression` as the classifier, with increased `max_iter` and `class_weight="balanced"`.
5. **Evaluation** on the held-out test set using the confusion matrix and `classification_report` (precision, recall, f1-score, accuracy).

## Results

On the test set, the logistic regression model achieves near-perfect performance on this dataset, with very high precision and recall for both churned and non-churned customers. The confusion matrix shows only a small number of misclassifications on 2,000 test samples, indicating that the model is able to separate the two classes very effectively on this data.

These results should be interpreted with some caution, as such high scores may indicate that the dataset is relatively easy to separate or that additional checks for data leakage and robustness (e.g. cross-validation, alternative splits) would be useful in a more production-oriented setting.

## Technologies

- Python 3
- pandas
- scikit-learn
- numpy
- Jupyter / Google Colab

## How to run

1. Clone the repository.
2. (Optional) Create and activate a virtual environment.
3. Install the required packages:

   ```bash
   pip install -r requirements.txt
