# stdlib
from pathlib import Path

# third party
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

current_dir = Path(__file__).parent


def load_adult_data(split_size=0.3):
    """
    > This function loads the adult dataset, removes all the rows with missing values, and then splits the data into
    a training and test set
    Args:
      split_size: The proportion of the dataset to include in the test split.
    Returns:
      X_train, X_test, y_train, y_test, X, y
    """

    def process_dataset(df):
        """
        > This function takes a dataframe, maps the categorical variables to numerical values, and returns a
        dataframe with the numerical values
        Args:
          df: The dataframe to be processed
        Returns:
          a dataframe after the mapping
        """

        data = [df]

        salary_map = {" <=50K": 1, " >50K": 0}
        df["salary"] = df["salary"].map(salary_map).astype(int)

        df["sex"] = df["sex"].map({" Male": 1, " Female": 0}).astype(int)

        df["country"] = df["country"].replace(" ?", np.nan)
        df["workclass"] = df["workclass"].replace(" ?", np.nan)
        df["occupation"] = df["occupation"].replace(" ?", np.nan)

        df.dropna(how="any", inplace=True)

        for dataset in data:
            dataset.loc[
                dataset["country"] != " United-States",
                "country",
            ] = "Non-US"
            dataset.loc[
                dataset["country"] == " United-States",
                "country",
            ] = "US"

        df["country"] = df["country"].map({"US": 1, "Non-US": 0}).astype(int)

        df["marital-status"] = df["marital-status"].replace(
            [
                " Divorced",
                " Married-spouse-absent",
                " Never-married",
                " Separated",
                " Widowed",
            ],
            "Single",
        )
        df["marital-status"] = df["marital-status"].replace(
            [" Married-AF-spouse", " Married-civ-spouse"],
            "Couple",
        )

        df["marital-status"] = df["marital-status"].map(
            {"Couple": 0, "Single": 1},
        )

        rel_map = {
            " Unmarried": 0,
            " Wife": 1,
            " Husband": 2,
            " Not-in-family": 3,
            " Own-child": 4,
            " Other-relative": 5,
        }

        df["relationship"] = df["relationship"].map(rel_map)

        race_map = {
            " White": 0,
            " Amer-Indian-Eskimo": 1,
            " Asian-Pac-Islander": 2,
            " Black": 3,
            " Other": 4,
        }

        df["race"] = df["race"].map(race_map)

        def f(x):
            if (
                x["workclass"] == " Federal-gov"
                or x["workclass"] == " Local-gov"
                or x["workclass"] == " State-gov"
            ):
                return "govt"
            elif x["workclass"] == " Private":
                return "private"
            elif (
                x["workclass"] == " Self-emp-inc"
                or x["workclass"] == " Self-emp-not-inc"
            ):
                return "self_employed"
            else:
                return "without_pay"

        df["employment_type"] = df.apply(f, axis=1)

        employment_map = {
            "govt": 0,
            "private": 1,
            "self_employed": 2,
            "without_pay": 3,
        }

        df["employment_type"] = df["employment_type"].map(employment_map)
        df.drop(
            labels=[
                "workclass",
                "education",
                "occupation",
            ],
            axis=1,
            inplace=True,
        )
        df.loc[(df["capital-gain"] > 0), "capital-gain"] = 1
        df.loc[(df["capital-gain"] == 0, "capital-gain")] = 0

        df.loc[(df["capital-loss"] > 0), "capital-loss"] = 1
        df.loc[(df["capital-loss"] == 0, "capital-loss")] = 0

        return df

    df = pd.read_csv(current_dir / "adult.csv", delimiter=",")

    df = process_dataset(df)

    df_sex_1 = df.query("sex ==1")

    X = df_sex_1.drop(["salary"], axis=1)
    y = df_sex_1["salary"]

    # Creation of Train and Test dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=split_size,
        random_state=42,
    )

    return X_train, X_test, y_train, y_test, X, y
