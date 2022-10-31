# Imports
# stdlib
from copy import deepcopy

# third party
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


def load_dataset(dataset, seed=None):
    """Function that loads the different datasets."""

    # third party
    from sklearn.model_selection import train_test_split

    if dataset == "support":
        """Load the Support dataset"""

        # Load csv
        df = pd.read_csv("../data/support_data.csv")

        df = df.drop(columns=["Unnamed: 0", "d.time"])

        # Assess feature-label correlations
        corr_matrix = df.corr()
        columns = list(np.abs(corr_matrix["death"]).sort_values(ascending=False).index)[
            ::-1
        ][0:-1]
        corr_vals = np.abs(corr_matrix["death"]).sort_values(ascending=False)[::-1][
            0:-1
        ]

        X = df.drop(columns=["death"])
        y = df["death"]

        # Get the column ids
        column_ids = []
        for name in columns:
            try:
                id = X.columns.get_loc(name)
                column_ids.append(id)
            except BaseException:
                pass
        column_ids = np.array(column_ids)

        # Split dataset
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

        # Make pandas dataset copy
        X_train_pd = deepcopy(X_train)
        X_test_pd = deepcopy(X_test)
        y_train_pd = deepcopy(y_train)
        y_test_pd = deepcopy(y_test)

        # Convert to numpy
        X_train = X_train.to_numpy().astype(np.float32)
        y_train = y_train.values.astype(np.float32)
        X_test = X_test.to_numpy().astype(np.float32)
        y_test = y_test.values.astype(np.float32)
        labels = np.unique(y_train)
        nlabels = len(labels)

        # Scale data
        # third party
        from sklearn import preprocessing

        scaler = preprocessing.StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

    if dataset == "covid":
        """Load the Covid dataset"""
        df = pd.read_csv("../data/covid.csv")
        x_ids = [
            2,
            3,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            20,
            21,
            22,
            23,
            24,
            25,
            26,
            27,
            28,
            29,
            30,
            31,
            32,
            33,
        ]
        X = df.iloc[:, x_ids]
        y = df.iloc[:, 0]

        # Assess feature-label correlations
        corr_matrix = df.corr()
        columns = list(
            np.abs(corr_matrix["is_dead"]).sort_values(ascending=False).index,
        )[::-1][0:-1]
        corr_vals = np.abs(corr_matrix["is_dead"]).sort_values(ascending=False)[::-1][
            0:-1
        ]

        # Get the column ids
        column_ids = []
        for name in columns:
            try:
                id = X.drop(columns=["Race", "SG_UF_NOT"]).columns.get_loc(name)
                column_ids.append(id)
            except BaseException:
                pass
        column_ids = np.array(column_ids)

        # Split dataset
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

        if seed is not None:
            # For specific example we want to show
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=0.4,
                random_state=seed,
            )

        # Make pandas dataset copy
        X_train_pd = deepcopy(X_train)
        X_test_pd = deepcopy(X_test)
        y_train_pd = deepcopy(y_train)
        y_test_pd = deepcopy(y_test)

        X_train = (
            X_train.drop(columns=["Race", "SG_UF_NOT"]).to_numpy().astype(np.float32)
        )
        y_train = y_train.values
        labels = np.unique(y_train)

        X_test = (
            X_test.drop(columns=["Race", "SG_UF_NOT"]).to_numpy().astype(np.float32)
        )
        y_test = y_test.values
        nlabels = len(labels)

    if dataset == "prostate":
        """Load the Prostate datasets"""
        df_feat_seer, df_label_seer, df = load_seer_cutract_dataset("seer", seed=42)
        df_feat_cutract, df_label_cutract, _ = load_seer_cutract_dataset(
            "cutract",
            seed=42,
        )

        # Combine datasets
        mid_cutract = int(len(df_feat_cutract) / 2)
        mid_seer = int(len(df_feat_seer) / 2)
        X_train = pd.concat(
            [df_feat_seer.iloc[0:mid_seer, :], df_feat_cutract.iloc[0:mid_cutract]],
            ignore_index=True,
        )
        y_train = pd.concat(
            [df_label_seer.iloc[0:mid_seer], df_label_cutract.iloc[0:mid_cutract]],
            ignore_index=True,
        )

        X_test = pd.concat(
            [
                df_feat_seer.iloc[mid_seer : len(df_feat_seer), :],
                df_feat_cutract.iloc[mid_cutract : len(df_feat_cutract), :],
            ],
            ignore_index=True,
        )
        y_test = pd.concat(
            [
                df_label_seer.iloc[mid_seer : len(df_feat_seer)],
                df_label_cutract.iloc[mid_cutract : len(df_feat_cutract)],
            ],
            ignore_index=True,
        )

        # Make pandas dataset copy
        X_test_pd = deepcopy(X_test)
        X_train_pd = deepcopy(X_train)
        y_test_pd = deepcopy(y_test)
        y_train_pd = deepcopy(y_train)

        # Convert to numpy
        X_train = X_train.to_numpy().astype(np.float32)
        y_train = y_train.values
        X_test = X_test.to_numpy().astype(np.float32)
        y_test = y_test.values

        labels = np.unique(y_train)
        nlabels = len(labels)
        df = deepcopy(X_train_pd)
        df["y"] = y_train_pd

        # Assess feature-label correlations
        corr_matrix = df.corr()
        columns = list(np.abs(corr_matrix["y"]).sort_values(ascending=False).index)[
            ::-1
        ][0:-1]
        corr_vals = np.abs(corr_matrix["y"]).sort_values(ascending=False)[::-1][0:-1]

        # Get the column ids
        column_ids = []
        for name in columns:
            try:
                id = X_train_pd.columns.get_loc(name)
                column_ids.append(id)
            except BaseException:
                pass
        column_ids = np.array(column_ids)

    if dataset == "fetal":
        """Load the Fetal dataset"""
        df = pd.read_excel("../data/CTG.xls", sheet_name="Raw Data")
        df = df.dropna()
        df.drop(
            columns=[
                "FileName",
                "Date",
                "SegFile",
                "A",
                "B",
                "C",
                "D",
                "E",
                "AD",
                "DE",
                "LD",
                "FS",
                "SUSP",
                "CLASS",
                "DR",
            ],
            inplace=True,
        )

        # Drop irrelevant columns
        df = df.drop(
            ["Median", "Mode", "Mean", "Width", "Min", "Max", "Nmax", "MSTV", "DL"],
            axis=1,
        )

        # class-idx relationship
        class2idx = {1: 0, 2: 1, 3: 2}

        df["NSP"].replace(class2idx, inplace=True)

        X = df.drop(["NSP"], axis=1)
        y = df["NSP"]

        # Split dataset
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

        # Make pandas dataset copy
        X_test_pd = deepcopy(X_test)
        X_train_pd = deepcopy(X_train)
        y_test_pd = deepcopy(y_test)
        y_train_pd = deepcopy(y_train)

        # Convert to numpy
        X_train = X_train.to_numpy().astype(np.float32)
        y_train = y_train.values
        labels = np.unique(y_train)
        nlabels = len(labels)

        # third party
        from sklearn import preprocessing

        # Scale data
        scaler = preprocessing.StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

        # Assess feature-label correlations
        corr_matrix = df.corr()
        columns = list(np.abs(corr_matrix["NSP"]).sort_values(ascending=False).index)[
            ::-1
        ][0:-1]
        corr_vals = np.abs(corr_matrix["NSP"]).sort_values(ascending=False)[::-1][0:-1]

        # Get the column ids
        column_ids = []
        for name in columns:
            try:
                id = X.columns.get_loc(name)
                column_ids.append(id)
            except BaseException:
                pass
        column_ids = np.array(column_ids)

    class TrainData(Dataset):
        def __init__(self, X_data, y_data):
            self.X_data = X_data
            self.y_data = y_data

        def __getitem__(self, index):
            return self.X_data[index], self.y_data[index]

        def __len__(self):
            return len(self.X_data)

    BATCH_SIZE = 128
    # Create train dataloader
    train_data = TrainData(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

    return (
        train_loader,
        train_data,
        X_train,
        y_train,
        X_test,
        y_test,
        X_train_pd,
        y_train_pd,
        X_test_pd,
        y_test_pd,
        nlabels,
        corr_vals,
        column_ids,
        df,
    )


def load_seer_cutract_dataset(name="seer", seed=42):
    """Function that loads the Seer and Cutract dataset."""

    # third party
    import pandas as pd
    import sklearn

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

    # features = [
    #     "age",
    #     "psa",
    #     "comorbidities",
    #     "treatment_CM",
    #     "treatment_Primary hormone therapy",
    #     "treatment_Radical Therapy-RDx",
    #     "treatment_Radical therapy-Sx",
    #     "grade",
    # ]

    # Features to keep
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
        n_samples = 15000
        ns = 10000
    else:
        n_samples = 5000
        ns = 1000
    df = pd.concat(
        [
            df_dead.sample(ns, random_state=seed),
            df_survive.sample(n_samples, random_state=seed),
        ],
    )

    df = sklearn.utils.shuffle(df, random_state=seed)
    df = df.reset_index(drop=True)
    return df[features], df[label], df
