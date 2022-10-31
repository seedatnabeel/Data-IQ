# stdlib
import sys

# third party
from catboost import CatBoostClassifier
from demo_data import load_adult_data
from lightgbm import LGBMClassifier
import pytest
import xgboost as xgb

# data_iq absolute
from data_iq.dataiq_class import DataIQ_SKLearn

X_train, X_test, y_train, y_test, X, y = load_adult_data(split_size=0.3)


def eval_helper(model, nest: int = 10, **kwargs):
    dataiq = DataIQ_SKLearn(X=X_train, y=y_train.to_numpy(), **kwargs)

    model.fit(X_train, y_train)

    for i in range(1, nest):
        dataiq.on_epoch_end(clf=model, iteration=i)

    aleatoric_uncertainty = dataiq.aleatoric
    confidence = dataiq.confidence

    assert len(X_train) == len(confidence)
    assert len(X_train) == len(aleatoric_uncertainty)

    assert max(confidence) <= 1
    assert max(aleatoric_uncertainty) <= 1


def test_xgboost_example() -> None:
    nest = 10
    clf = xgb.XGBClassifier(n_estimators=nest)

    eval_helper(clf, nest=nest)


@pytest.mark.skipif(sys.platform == "darwin", reason="LGBM crash on OSX")
def test_lgm_example() -> None:
    nest = 10

    clf = LGBMClassifier(n_estimators=nest)
    eval_helper(clf, nest=nest)


def test_catboost_example() -> None:
    nest = 10

    clf = CatBoostClassifier(n_estimators=nest)
    eval_helper(clf, nest=nest, catboost=True)
