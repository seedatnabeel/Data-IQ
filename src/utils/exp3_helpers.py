# stdlib
import sys

# third party
import numpy as np
import pandas as pd
import sklearn
import xgboost as xgb

sys.path.insert(0, "../..")

# data_iq absolute
from data_iq.dataiq_class import DataIQ_SKLearn  # noqa: E402


def load_seer_cutract_dataset(name, seed):
    """
    It loads the SEER/CUTRACT dataset, and returns the features, labels, and the entire dataset

    Args:
      name: the name of the dataset to load.
      seed: the random seed used to generate the data

    Returns:
      The features, labels, and the entire dataset.
    """

    def aggregate_grade(row):
        if row["grade_1.0"] == 1:
            return 1
        if row["grade_2.0"] == 1:
            return 2
        if row["grade_3.0"] == 1:
            return 3
        if row["grade_4.0"] == 1:
            return 4
        if row["grade_5.0"] == 1:
            return 5

    def aggregate_stage(row):
        if row["stage_1"] == 1:
            return 1
        if row["stage_2"] == 1:
            return 2
        if row["stage_3"] == 1:
            return 3
        if row["stage_4"] == 1:
            return 4
        if row["stage_5"] == 1:
            return 5

    random_seed = seed

    features = [
        "age",
        "psa",
        "comorbidities",
        "treatment_CM",
        "treatment_Primary hormone therapy",
        "treatment_Radical Therapy-RDx",
        "treatment_Radical therapy-Sx",
        "grade",
    ]

    features = [
        "age",
        "psa",
        "comorbidities",
        "treatment_CM",
        "treatment_Primary hormone therapy",
        "treatment_Radical Therapy-RDx",
        "treatment_Radical therapy-Sx",
        "grade",
        "stage_1",
        "stage_2",
        "stage_3",
        "stage_4",
    ]

    label = "mortCancer"
    df = pd.read_csv(f"../data/{name}.csv")

    df["grade"] = df.apply(aggregate_grade, axis=1)
    df["stage"] = df.apply(aggregate_stage, axis=1)
    df["mortCancer"] = df["mortCancer"].astype(int)
    df["mort"] = df["mort"].astype(int)

    mask = df[label] == True  # noqa: E712
    df_dead = df[mask]
    df_survive = df[~mask]

    if name == "seer":
        n_samples = 1000
        ns = 1000
    else:
        n_samples = 1000
        ns = 1000
    df = pd.concat(
        [
            df_dead.sample(ns, random_state=random_seed),
            df_survive.sample(n_samples, random_state=random_seed),
        ],
    )
    df = sklearn.utils.shuffle(df, random_state=random_seed)
    df = df.reset_index(drop=True)
    return df[features], df[label], df


def get_data(dataset, seed=42):
    """
    It takes in the training data and labels, and the number of estimators, and returns the indices of
    the easy, inconsistent, and hard training examples.

    Args:
      dataset: the name of the dataset you want to use.
      seed: the random seed used to split the data into train and test sets. Defaults to 42
    """
    df_feat, df_label, df = load_seer_cutract_dataset(dataset, seed=seed)

    mid = int(len(df_feat) / 2)

    X_train = df_feat.iloc[0:mid]
    y_train = df_label.iloc[0:mid]

    X_test = df_feat.iloc[mid : len(df_feat), :]
    y_test = df_label.iloc[mid : len(df_feat)]

    return X_train, y_train, X_test, y_test


def filter_with_dataiq(X_train, y_train, nest):
    """
    It takes in the training data and labels, and the number of trees in the XGBoost model, and returns
    the indices of the easy, ambiguous, and hard training examples, as well as the aleatoric uncertainty
    of each training example

    Args:
      X_train: the training data
      y_train: the labels for the training data
      nest: number of estimators

    Returns:
      the indices of the easy, ambiguous, and hard training examples.
    """

    # Train the XGBoost model
    clf = xgb.XGBClassifier(n_estimators=nest)
    clf.fit(X_train, y_train)

    # Initialize the DataIQ object
    dataiq = DataIQ_SKLearn(X=X_train, y=y_train)

    # Compute DataIQ
    for i in range(1, nest):
        dataiq.on_epoch_end(clf=clf, iteration=i)

    # Get metrics
    aleatoric_train = dataiq.aleatoric
    confidence_train = dataiq.confidence

    percentile_thresh = 50
    conf_thresh_low = 0.25
    conf_thresh_high = 0.75

    # Get the 3 subgroups
    hard_train = np.where(
        (confidence_train <= conf_thresh_low)
        & (aleatoric_train <= np.percentile(aleatoric_train, percentile_thresh)),
    )[0]
    easy_train = np.where(
        (confidence_train >= conf_thresh_high)
        & (aleatoric_train <= np.percentile(aleatoric_train, percentile_thresh)),
    )[0]

    hard_easy = np.concatenate((hard_train, easy_train))
    ambig_train = []
    for id in range(len(confidence_train)):
        if id not in hard_easy:
            ambig_train.append(id)
    ambig_train = np.array(ambig_train)

    filtered_ids = np.concatenate((easy_train, ambig_train))  # filtered ids
    return easy_train, ambig_train, hard_train, aleatoric_train, filtered_ids
